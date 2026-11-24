//! MPRIS media player control via direct D-Bus.
//!
//! Pauses playing players before recording starts and resumes only the
//! players we actually paused as soon as capture stops. The previous
//! implementation shelled out to `playerctl`, which had two failure modes
//! that hit real users:
//!
//!   * `playerctl -l` silently filters out some MPRIS-compliant players
//!     (e.g. `cliamp`) even when they expose a complete MPRIS interface
//!     on the bus, so voxtype never tried to pause them.
//!   * The resume path called `playerctl --player <stored-name> play`
//!     and ignored the exit code. If the player's bus name had gone
//!     away during the dictation window (Chromium's PID-suffixed names
//!     are particularly fragile), the resume silently no-opped and the
//!     user's music stayed paused. See Omarchy issue #6029.
//!
//! Talking D-Bus directly via zbus fixes both: we enumerate all owned
//! names matching `org.mpris.MediaPlayer2.*` ourselves and surface real
//! errors on resume.
//!
//! A headset that switches to HSP/HFP for the microphone needs one more
//! step. `Play` sent the instant capture stops lands while A2DP is still
//! coming back, the transport errors, and the player stays paused. Resume
//! keeps pressing Play for a few seconds once the headset is on A2DP again.

/// PulseAudio/PipeWire stream volume captured before media ducking.
///
/// Only the Linux `imp` module reads these, but `Daemon` holds a
/// `Vec<DuckedMediaStream>` on every platform, so the type cannot be gated.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct DuckedMediaStream {
    index: u32,
    volumes: Vec<String>,
}

#[cfg(target_os = "linux")]
mod imp {
    use super::DuckedMediaStream;
    use serde_json::Value;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};
    use tokio::process::Command;
    use tokio::task::JoinHandle;
    use tokio::time::sleep;
    use tracing::{debug, warn};
    use zbus::{fdo::DBusProxy, Connection, Proxy};

    /// Bumped when a recording pauses players. A resume copies this value
    /// when it is created and stops if the value changes, so Play cannot
    /// land under the next dictation. The copy happens before the resume
    /// is spawned: the task may not run until after that next pause.
    static MEDIA_RESUME_EPOCH: AtomicU64 = AtomicU64::new(0);

    fn bump_media_resume_epoch() -> u64 {
        MEDIA_RESUME_EPOCH.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn media_resume_cancelled(epoch: u64) -> bool {
        MEDIA_RESUME_EPOCH.load(Ordering::Relaxed) != epoch
    }

    const MPRIS_PREFIX: &str = "org.mpris.MediaPlayer2.";
    const MPRIS_PATH: &str = "/org/mpris/MediaPlayer2";
    const MPRIS_IFACE: &str = "org.mpris.MediaPlayer2.Player";

    /// Pause all currently playing MPRIS media players.
    /// Returns the bus names of players that were paused so they can be resumed.
    /// Suffixes in `ignored` are matched against the part after the MPRIS prefix
    /// either exactly or as a `<entry>.<instance>` prefix (so `"chromium"`
    /// matches `chromium.instance1872063`).
    pub async fn pause_playing_players(ignored: &[String]) -> Vec<String> {
        // A resume may still be waiting for the headset to return to A2DP.
        let _ = bump_media_resume_epoch();
        let conn = match Connection::session().await {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to connect to session bus: {e}");
                return Vec::new();
            }
        };

        let players = match list_mpris_players(&conn, ignored).await {
            Ok(p) => p,
            Err(e) => {
                warn!("Failed to enumerate MPRIS players: {e}");
                return Vec::new();
            }
        };

        if players.is_empty() {
            debug!("No MPRIS players found");
            return Vec::new();
        }

        let mut paused = Vec::new();
        for bus_name in players {
            match player_status(&conn, &bus_name).await {
                Ok(status) if status == "Playing" => {
                    debug!(player = %bus_name, "Pausing media player");
                    match call_player(&conn, &bus_name, "Pause").await {
                        Ok(()) => paused.push(bus_name),
                        Err(e) => warn!(player = %bus_name, "Pause failed: {e}"),
                    }
                }
                Ok(status) => {
                    debug!(player = %bus_name, %status, "Skipping non-playing player")
                }
                Err(e) => debug!(player = %bus_name, "Status query failed: {e}"),
            }
        }

        if !paused.is_empty() {
            debug!("Paused {} media player(s)", paused.len());
        }
        paused
    }

    /// Resume previously-paused MPRIS media players.
    ///
    /// The first Play goes out immediately so a speaker sink starts with the
    /// mic. A Bluetooth headset is then given a few seconds to finish
    /// switching back to A2DP, and Play is pressed again if the first one
    /// did not stick.
    ///
    /// The epoch is read before this future is awaited or spawned. A later
    /// pause bumps it and this resume will not press Play.
    pub fn resume_players(players: Vec<String>) -> impl std::future::Future<Output = ()> + Send {
        let epoch = MEDIA_RESUME_EPOCH.load(Ordering::Relaxed);
        async move {
            if players.is_empty() {
                return;
            }
            let conn = match Connection::session().await {
                Ok(c) => c,
                Err(e) => {
                    warn!("Failed to connect to session bus for resume: {e}");
                    return;
                }
            };

            if !press_play_unless_playing(&conn, &players, epoch).await {
                debug!("No media players left to resume");
                return;
            }
            settle_resumed_playback(&conn, &players, epoch).await;
        }
    }

    /// How long to keep trying after the microphone closes. Long enough for
    /// a headset to leave HSP/HFP and for the A2DP transport to survive the
    /// error that follows the switch, short enough to give up if the player
    /// is gone.
    const RESUME_SETTLE: Duration = Duration::from_secs(12);

    /// Playback that shows up before this, on a headset, is still the Play
    /// that raced the profile switch. It usually dies with the transport.
    const BLUETOOTH_TRANSPORT_SETTLE: Duration = Duration::from_secs(2);

    /// Press Play on players that are not already playing.
    ///
    /// Returns false when every named player has left the bus, or when a new
    /// recording cancelled this resume.
    async fn press_play_unless_playing(conn: &Connection, players: &[String], epoch: u64) -> bool {
        let mut any = false;
        for bus_name in players {
            if media_resume_cancelled(epoch) {
                return false;
            }
            match player_status(conn, bus_name).await {
                Ok(status) if status == "Playing" => any = true,
                Ok(_) => {
                    any = true;
                    debug!(player = %bus_name, "Resuming media player");
                    if let Err(e) = call_player(conn, bus_name, "Play").await {
                        warn!(
                            player = %bus_name,
                            "Resume failed (player may have exited during dictation): {e}"
                        );
                    }
                }
                Err(e) => debug!(player = %bus_name, "Status query failed: {e}"),
            }
        }
        any
    }

    /// Keep checking until resumed playback survives the headset switch.
    async fn settle_resumed_playback(conn: &Connection, players: &[String], epoch: u64) {
        let started = Instant::now();
        let deadline = started + RESUME_SETTLE;
        let mut saw_route = false;
        while Instant::now() < deadline {
            if media_resume_cancelled(epoch) {
                debug!("Media resume cancelled");
                return;
            }
            let route = match current_audio_route().await {
                Ok(route) => route,
                Err(e) => {
                    debug!("Media route unavailable while resuming: {e}");
                    sleep(Duration::from_millis(250)).await;
                    continue;
                }
            };
            saw_route = true;
            if !route.ready {
                sleep(Duration::from_millis(250)).await;
                continue;
            }
            // The profile can flap. Wait one beat and read it again.
            sleep(Duration::from_millis(400)).await;
            if media_resume_cancelled(epoch) {
                return;
            }
            let route = match current_audio_route().await {
                Ok(route) if route.ready => route,
                _ => continue,
            };
            if !press_play_unless_playing(conn, players, epoch).await {
                debug!("No media players left to resume");
                return;
            }
            sleep(Duration::from_secs(1)).await;
            if media_resume_cancelled(epoch) {
                return;
            }
            if resumed_playback_held(conn, players, &route).await {
                // On a headset, a Playing status in the first two seconds is
                // the Play that races the transport error. Do not treat it
                // as settled, and do not pause it just to nudge.
                if route.bluetooth && started.elapsed() < BLUETOOTH_TRANSPORT_SETTLE {
                    continue;
                }
                sleep(Duration::from_millis(1200)).await;
                if media_resume_cancelled(epoch) {
                    return;
                }
                let route = match current_audio_route().await {
                    Ok(route) if route.ready => route,
                    _ => continue,
                };
                if resumed_playback_held(conn, players, &route).await {
                    debug!("Resumed media stayed playing");
                    return;
                }
            }
            debug!("Media resume did not stick, trying again");
            if media_resume_cancelled(epoch) {
                return;
            }
            nudge_players(conn, players, epoch).await;
        }
        if saw_route {
            warn!("Gave up waiting for media to resume");
        } else {
            debug!("Media resume settle skipped; pactl could not describe the default sink");
        }
    }

    /// Pause anything that claims to be playing, then press Play.
    ///
    /// A player left at Playing with no stream ignores a second Play. The
    /// pause makes the next Play a real transition.
    async fn nudge_players(conn: &Connection, players: &[String], epoch: u64) {
        for bus_name in players {
            if media_resume_cancelled(epoch) {
                return;
            }
            if player_status(conn, bus_name).await.ok().as_deref() == Some("Playing") {
                let _ = call_player(conn, bus_name, "Pause").await;
            }
            if media_resume_cancelled(epoch) {
                return;
            }
            let _ = call_player(conn, bus_name, "Play").await;
        }
    }

    /// True when every player still on the bus is Playing, and a headset
    /// sink actually has someone else's stream on it.
    async fn resumed_playback_held(
        conn: &Connection,
        players: &[String],
        route: &AudioRoute,
    ) -> bool {
        let mut any = false;
        for bus_name in players {
            match player_status(conn, bus_name).await {
                Ok(status) if status == "Playing" => any = true,
                Ok(_) => return false,
                Err(e) => debug!(player = %bus_name, "Status query failed: {e}"),
            }
        }
        if !any {
            return false;
        }
        if !route.bluetooth {
            return true;
        }
        let Some(sink_index) = route.sink_index else {
            return false;
        };
        bluetooth_has_foreign_input(sink_index).await
    }

    async fn bluetooth_has_foreign_input(sink_index: u32) -> bool {
        let (inputs, clients) = tokio::join!(pactl_list("sink-inputs"), pactl_list("clients"));
        let Ok(inputs) = inputs else {
            return false;
        };
        let clients = clients.unwrap_or(Value::Null);
        sink_has_foreign_input(sink_index, &inputs, &clients, std::process::id())
    }

    async fn current_audio_route() -> Result<AudioRoute, String> {
        let output = Command::new("pactl")
            .args(["get-default-sink"])
            .output()
            .await
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(format!(
                "pactl get-default-sink exited with {}",
                output.status
            ));
        }
        let name = String::from_utf8_lossy(&output.stdout);
        let (sinks, cards) = tokio::join!(pactl_list("sinks"), pactl_list("cards"));
        Ok(audio_route(name.trim(), &sinks?, &cards?))
    }

    /// Where music should come back, and whether that sink can take it yet.
    struct AudioRoute {
        bluetooth: bool,
        ready: bool,
        sink_index: Option<u32>,
    }

    /// `default_sink` is `pactl get-default-sink`. A `bluez_output` sink is
    /// ready only once its card is on an A2DP profile and a sink for that
    /// address exists. Any other sink is ready immediately.
    fn audio_route(default_sink: &str, sinks: &Value, cards: &Value) -> AudioRoute {
        if let Some((card, prefix)) = bluetooth_identity(default_sink) {
            let profile = card_active_profile(cards, &card).unwrap_or("");
            let on_a2dp = profile.starts_with("a2dp");
            let sink_index = if on_a2dp {
                sink_index_matching(sinks, |name| name == default_sink)
                    .or_else(|| sink_index_matching(sinks, |name| name.starts_with(&prefix)))
            } else {
                None
            };
            return AudioRoute {
                bluetooth: true,
                ready: sink_index.is_some(),
                sink_index,
            };
        }
        AudioRoute {
            bluetooth: false,
            ready: true,
            sink_index: sink_index_matching(sinks, |name| name == default_sink),
        }
    }

    /// `bluez_output.ADDR.N` -> card `bluez_card.ADDR` and sink prefix
    /// `bluez_output.ADDR.`.
    fn bluetooth_identity(sink_name: &str) -> Option<(String, String)> {
        let rest = sink_name.strip_prefix("bluez_output.")?;
        let (addr, _) = rest.rsplit_once('.')?;
        if addr.is_empty() {
            return None;
        }
        Some((
            format!("bluez_card.{addr}"),
            format!("bluez_output.{addr}."),
        ))
    }

    fn card_active_profile<'a>(cards: &'a Value, card_name: &str) -> Option<&'a str> {
        cards.as_array()?.iter().find_map(|card| {
            if card.get("name").and_then(Value::as_str) == Some(card_name) {
                card.get("active_profile").and_then(Value::as_str)
            } else {
                None
            }
        })
    }

    fn sink_index_matching(sinks: &Value, predicate: impl Fn(&str) -> bool) -> Option<u32> {
        sinks.as_array()?.iter().find_map(|sink| {
            let name = sink.get("name").and_then(Value::as_str)?;
            if predicate(name) {
                sink.get("index").and_then(value_as_u32)
            } else {
                None
            }
        })
    }

    fn sink_has_foreign_input(
        sink_index: u32,
        inputs: &Value,
        clients: &Value,
        own_pid: u32,
    ) -> bool {
        let Some(streams) = inputs.as_array() else {
            return false;
        };
        streams.iter().any(|stream| {
            stream.get("sink").and_then(value_as_u32) == Some(sink_index)
                && !stream.get("mute").and_then(Value::as_bool).unwrap_or(false)
                && !stream
                    .get("corked")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                && !is_voxtype_stream(stream, clients, own_pid)
        })
    }

    /// Interval between ramp steps. Short enough to sound continuous, long
    /// enough to keep the number of `pactl` invocations per fade small.
    const FADE_STEP_MS: u64 = 20;

    /// Intermediate scaling factors for a fade, excluding the final target.
    ///
    /// The caller sets `to_percent` itself, which keeps the exact target and
    /// its error handling in a single place. Returns an empty vector when the
    /// fade is too short to produce a step, so `fade_ms = 0` means "instant".
    fn ramp_factors(from_percent: u8, to_percent: u8, fade_ms: u32) -> Vec<u8> {
        let steps = u64::from(fade_ms).div_ceil(FADE_STEP_MS);
        if steps <= 1 {
            return Vec::new();
        }

        let from = f32::from(from_percent);
        let span = f32::from(to_percent) - from;
        (1..steps)
            .map(|step| {
                let factor = from + span * (step as f32 / steps as f32);
                factor.round().clamp(0.0, f32::from(u8::MAX)) as u8
            })
            .collect()
    }

    /// Walk `streams` from `from_percent` towards `to_percent` over `fade_ms`,
    /// stopping short of the target — the caller writes that itself, which
    /// keeps the exact value and its error handling in one place.
    ///
    /// Best effort: a failed intermediate step is ignored, since the caller's
    /// final write is authoritative.
    async fn ramp_streams(
        streams: &[DuckedMediaStream],
        from_percent: u8,
        to_percent: u8,
        fade_ms: u32,
    ) {
        for factor in ramp_factors(from_percent, to_percent, fade_ms) {
            for stream in streams {
                let _ = Command::new("pactl")
                    .arg("set-sink-input-volume")
                    .arg(stream.index.to_string())
                    .args(scaled_volumes(&stream.volumes, factor))
                    .status()
                    .await;
            }
            tokio::time::sleep(std::time::Duration::from_millis(FADE_STEP_MS)).await;
        }
    }

    /// Set every stream to `amplitude_percent` of its original volume.
    async fn apply_volumes(streams: &[DuckedMediaStream], amplitude_percent: u8) {
        for stream in streams {
            let target = scaled_volumes(&stream.volumes, amplitude_percent);
            debug!(
                stream = stream.index,
                from = ?stream.volumes,
                to = ?target,
                factor_percent = amplitude_percent,
                "Ducking media stream"
            );
            match Command::new("pactl")
                .arg("set-sink-input-volume")
                .arg(stream.index.to_string())
                .args(&target)
                .status()
                .await
            {
                Ok(status) if status.success() => {}
                Ok(status) => warn!(
                    stream = stream.index,
                    "Media ducking failed with status: {status}"
                ),
                Err(e) => warn!(
                    stream = stream.index,
                    "Failed to run pactl for media ducking: {e}"
                ),
            }
        }
    }

    /// Lower active sink-input volumes and return their original volumes.
    ///
    /// This intentionally does not use MPRIS transport controls. It can be
    /// enabled alongside `pause_media`, but when both are enabled the pause
    /// feature keeps its existing start/stop behavior and ducking only manages
    /// stream volume.
    ///
    /// The originals are captured before anything is written, and the fade
    /// itself runs on a spawned task so the caller can open the microphone
    /// without waiting out `fade_ms`. The returned handle is that task: the
    /// caller must await it before capturing originals again, or a second
    /// recording started mid-fade would record intermediate volumes as the new
    /// "originals" and media would drift permanently quieter.
    pub async fn duck_playing_audio(
        volume_percent: u8,
        fade_ms: u32,
    ) -> (Vec<DuckedMediaStream>, Option<JoinHandle<()>>) {
        let streams = match list_active_sink_inputs().await {
            Ok(streams) => streams,
            Err(e) => {
                warn!("Failed to enumerate audio streams for media ducking: {e}");
                return (Vec::new(), None);
            }
        };

        if streams.is_empty() {
            debug!("No active audio streams found for media ducking");
            return (Vec::new(), None);
        }

        let factor = volume_percent.min(150);
        let fading = streams.clone();
        let handle = tokio::spawn(async move {
            ramp_streams(&fading, 100, factor, fade_ms).await;
            apply_volumes(&fading, factor).await;
        });

        (streams, Some(handle))
    }

    /// Restore stream volumes captured by `duck_playing_audio`.
    pub async fn restore_ducked_audio(
        streams: Vec<DuckedMediaStream>,
        ducked_percent: u8,
        fade_ms: u32,
    ) {
        if streams.is_empty() {
            return;
        }

        ramp_streams(&streams, ducked_percent.min(150), 100, fade_ms).await;

        let total = streams.len();
        let mut restored = 0usize;
        for stream in streams {
            debug!(stream = stream.index, "Restoring ducked media stream");
            match Command::new("pactl")
                .arg("set-sink-input-volume")
                .arg(stream.index.to_string())
                .args(&stream.volumes)
                .status()
                .await
            {
                Ok(status) if status.success() => restored += 1,
                Ok(status) => warn!(
                    stream = stream.index,
                    "Media duck restore failed with status: {status}"
                ),
                Err(e) => warn!(
                    stream = stream.index,
                    "Failed to run pactl for media duck restore: {e}"
                ),
            }
        }

        debug!("Restored {restored}/{total} ducked media stream(s)");
    }

    async fn list_active_sink_inputs() -> Result<Vec<DuckedMediaStream>, String> {
        // The client list only helps recognise voxtype's own streams, so it
        // is fetched alongside the streams, not after them, and recording
        // start waits no longer than it did for one `pactl` call.
        let (streams, clients) = tokio::join!(pactl_list("sink-inputs"), pactl_list("clients"));
        let clients = clients.unwrap_or_else(|e| {
            debug!("Failed to list audio clients, matching own streams by name only: {e}");
            Value::Null
        });
        Ok(parse_sink_inputs(&streams?, &clients, std::process::id()))
    }

    async fn pactl_list(what: &str) -> Result<Value, String> {
        let output = Command::new("pactl")
            .args(["-f", "json", "list", what])
            .output()
            .await
            .map_err(|e| e.to_string())?;

        if !output.status.success() {
            return Err(format!("pactl exited with {}", output.status));
        }

        serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())
    }

    /// Active sink inputs that ducking may lower: everything playing except
    /// voxtype's own streams. `clients` is `pactl -f json list clients`, or
    /// `Value::Null` when that could not be read.
    fn parse_sink_inputs(value: &Value, clients: &Value, own_pid: u32) -> Vec<DuckedMediaStream> {
        let Some(streams) = value.as_array() else {
            return Vec::new();
        };

        streams
            .iter()
            .filter(|stream| {
                !stream.get("mute").and_then(Value::as_bool).unwrap_or(false)
                    && !stream
                        .get("corked")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
            })
            .filter(|stream| {
                let own = is_voxtype_stream(stream, clients, own_pid);
                if own {
                    debug!(
                        stream = ?stream.get("index"),
                        "Leaving voxtype's own audio stream out of media ducking"
                    );
                }
                !own
            })
            .filter_map(|stream| {
                let index = value_as_u32(stream.get("index")?)?;
                let volumes = stream_volumes(stream);
                if volumes.is_empty() {
                    None
                } else {
                    Some(DuckedMediaStream { index, volumes })
                }
            })
            .collect()
    }

    /// Whether a sink input is played by voxtype itself.
    ///
    /// The daemon's cue sounds go out through a stream it keeps open, so that
    /// stream is active when recording starts and used to be ducked with the
    /// media, which left the stop and completion cues playing quietly. Any
    /// voxtype process counts, not only this one: none of them is the media
    /// the user asked to have turned down.
    ///
    /// The stream alone often cannot say which process owns it. cpal plays
    /// through PipeWire's ALSA plugin, whose streams carry no
    /// `application.process.id`, only the binary's name inside
    /// `application.name` and `node.name`. The process id and binary are on
    /// the client the stream's `client` field points at, so both are checked.
    fn is_voxtype_stream(stream: &Value, clients: &Value, own_pid: u32) -> bool {
        let client = stream
            .get("client")
            .and_then(value_as_u32)
            .and_then(|index| {
                clients
                    .as_array()?
                    .iter()
                    .find(|client| client.get("index").and_then(value_as_u32) == Some(index))
            });

        [Some(stream), client]
            .into_iter()
            .flatten()
            .filter_map(|object| object.get("properties"))
            .any(|properties| owned_by_voxtype(properties, own_pid))
    }

    /// Whether a stream's or client's properties name this process or a
    /// voxtype binary.
    fn owned_by_voxtype(properties: &Value, own_pid: u32) -> bool {
        let text = |key: &str| properties.get(key).and_then(Value::as_str);

        let is_this_process = ["application.process.id", "pipewire.sec.pid"]
            .into_iter()
            .any(|key| properties.get(key).and_then(value_as_u32) == Some(own_pid));

        // PipeWire's ALSA plugin names its streams after the program:
        // "PipeWire ALSA [voxtype]" and "alsa_playback.voxtype".
        let alsa_program = text("application.name")
            .and_then(|name| name.strip_prefix("PipeWire ALSA [")?.strip_suffix(']'))
            .or_else(|| text("node.name")?.strip_prefix("alsa_playback."));

        is_this_process
            || [text("application.process.binary"), alsa_program]
                .into_iter()
                .flatten()
                .any(is_voxtype_binary)
    }

    /// Whether a program name is one of voxtype's binaries: `voxtype`, a
    /// variant such as `voxtype-onnx-cuda-13`, or a Nix wrapper's
    /// `.voxtype-wrapped`.
    fn is_voxtype_binary(name: &str) -> bool {
        name.trim_start_matches('.').starts_with("voxtype")
    }

    fn value_as_u32(value: &Value) -> Option<u32> {
        value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .or_else(|| value.as_str()?.parse().ok())
    }

    fn stream_volumes(stream: &Value) -> Vec<String> {
        let Some(volume) = stream.get("volume").and_then(Value::as_object) else {
            return Vec::new();
        };
        let mut channels: Vec<_> = volume.iter().collect();
        channels.sort_by(|a, b| a.0.cmp(b.0));
        channels
            .into_iter()
            .filter_map(|(_, channel)| {
                channel
                    .get("value_percent")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .collect()
    }

    /// Scale each channel's `value_percent` so the stream keeps
    /// `amplitude_percent` of its current amplitude.
    ///
    /// PulseAudio's percentage scale is cubic (`amplitude = (percent /
    /// 100)^3`), so keeping half the amplitude does NOT mean halving the
    /// percentage: the percentage only shrinks by the cube root of the
    /// requested fraction. Scaling the percentage directly instead — the
    /// previous behaviour — cubed the reduction, so a configured 50 left
    /// 12.5% of the amplitude (-18 dB) and 30 was effectively mute.
    fn scaled_volumes(volumes: &[String], amplitude_percent: u8) -> Vec<String> {
        let gain = (f32::from(amplitude_percent) / 100.0).cbrt();
        volumes
            .iter()
            .map(|volume| {
                let numeric = volume.trim().trim_end_matches('%');
                let Ok(value) = numeric.parse::<f32>() else {
                    return volume.clone();
                };
                let scaled = value * gain;
                format!("{scaled:.0}%")
            })
            .collect()
    }

    async fn list_mpris_players(
        conn: &Connection,
        ignored: &[String],
    ) -> zbus::Result<Vec<String>> {
        let dbus = DBusProxy::new(conn).await?;
        let names = dbus.list_names().await?;
        let mut out = Vec::new();
        for n in names {
            let s: &str = n.as_str();
            if !s.starts_with(MPRIS_PREFIX) {
                continue;
            }
            let suffix = &s[MPRIS_PREFIX.len()..];
            // Skip playerctld's aggregator: pausing it would double-fire
            // pause across every underlying player.
            if suffix == "playerctld" {
                continue;
            }
            if ignored.iter().any(|ig| {
                suffix == ig
                    || suffix.starts_with(ig) && suffix.as_bytes().get(ig.len()) == Some(&b'.')
            }) {
                debug!(player = %suffix, "Ignored by config");
                continue;
            }
            out.push(s.to_string());
        }
        Ok(out)
    }

    async fn player_status(conn: &Connection, bus_name: &str) -> zbus::Result<String> {
        let proxy = Proxy::new(conn, bus_name, MPRIS_PATH, MPRIS_IFACE).await?;
        proxy.get_property::<String>("PlaybackStatus").await
    }

    async fn call_player(
        conn: &Connection,
        bus_name: &str,
        method: &'static str,
    ) -> zbus::Result<()> {
        let proxy = Proxy::new(conn, bus_name, MPRIS_PATH, MPRIS_IFACE).await?;
        proxy.call::<_, _, ()>(method, &()).await
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The cube-root correction changed what `duck_media_volume_percent`
        /// means, so the shipped default had to move with it or ducking would
        /// have quietly become nearly inaudible: the old default of 70 was
        /// applied to PulseAudio's cubic scale and actually left 0.70^3 of the
        /// amplitude. Pin that the new default reproduces the old depth, since
        /// a well-meaning "round it back up to 70" would silently undo it.
        #[test]
        fn default_duck_percent_preserves_pre_correction_loudness() {
            let old_default_amplitude = 0.70_f32.powi(3);
            let new_default_amplitude =
                f32::from(crate::config::AudioConfig::default().duck_media_volume_percent) / 100.0;

            assert!(
                (new_default_amplitude - old_default_amplitude).abs() < 0.01,
                "default duck depth drifted: {new_default_amplitude} vs {old_default_amplitude}"
            );
        }

        #[tokio::test]
        async fn list_skips_non_mpris_and_playerctld() {
            // We can't easily fake a real session bus, but we can at least
            // exercise the filter logic by running against the live bus and
            // confirming playerctld and ignored entries never appear.
            let Ok(conn) = Connection::session().await else {
                eprintln!("skip: no session bus");
                return;
            };
            let players = list_mpris_players(&conn, &["chromium".to_string()])
                .await
                .unwrap_or_default();
            for p in players {
                assert!(p.starts_with(MPRIS_PREFIX));
                let suffix = &p[MPRIS_PREFIX.len()..];
                assert_ne!(suffix, "playerctld");
                assert!(
                    !(suffix == "chromium" || suffix.starts_with("chromium.")),
                    "ignored prefix leaked: {p}"
                );
            }
        }

        #[test]
        fn ramp_is_skipped_when_the_fade_is_shorter_than_one_step() {
            assert!(ramp_factors(100, 55, 0).is_empty());
            assert!(ramp_factors(100, 55, FADE_STEP_MS as u32).is_empty());
        }

        #[test]
        fn ramp_walks_down_towards_the_target_without_reaching_it() {
            // 100 ms at a 20 ms step is 5 slices, so 4 intermediate factors;
            // the exact target is set by the caller, not by the ramp.
            let factors = ramp_factors(100, 50, 100);
            assert_eq!(factors, vec![90, 80, 70, 60]);
        }

        #[test]
        fn ramp_walks_up_when_restoring() {
            let factors = ramp_factors(50, 100, 100);
            assert_eq!(factors, vec![60, 70, 80, 90]);
        }

        #[test]
        fn ramp_stays_monotonic_and_inside_the_endpoints() {
            let factors = ramp_factors(100, 33, 250);
            assert!(!factors.is_empty());
            for pair in factors.windows(2) {
                assert!(pair[1] <= pair[0], "not monotonic: {factors:?}");
            }
            assert!(factors.iter().all(|&f| (33..=100).contains(&f)));
        }

        #[test]
        fn parses_active_sink_inputs_with_channel_volumes() {
            let value: Value = serde_json::json!([
                {
                    "index": 42,
                    "mute": false,
                    "corked": false,
                    "volume": {
                        "front-right": { "value_percent": "80%" },
                        "front-left": { "value_percent": "75%" }
                    }
                },
                {
                    "index": 43,
                    "mute": true,
                    "corked": false,
                    "volume": {
                        "mono": { "value_percent": "100%" }
                    }
                }
            ]);

            let streams = parse_sink_inputs(&value, &Value::Null, 1);
            assert_eq!(streams.len(), 1);
            assert_eq!(streams[0].index, 42);
            assert_eq!(streams[0].volumes, vec!["75%", "80%"]);
        }

        /// A playing sink input with the given `client` and properties.
        fn sink_input(index: u32, client: &str, properties: Value) -> Value {
            serde_json::json!({
                "index": index,
                "client": client,
                "mute": false,
                "corked": false,
                "volume": { "mono": { "value_percent": "100%" } },
                "properties": properties
            })
        }

        fn ducked_indexes(streams: &Value, clients: &Value, own_pid: u32) -> Vec<u32> {
            parse_sink_inputs(streams, clients, own_pid)
                .iter()
                .map(|stream| stream.index)
                .collect()
        }

        /// The stream the daemon plays its cues through, as `pactl` reported
        /// it on a PipeWire desktop: no process id anywhere on the stream,
        /// only the binary's name. It was ducked along with the media, so the
        /// stop and completion cues played quietly.
        #[test]
        fn own_cue_stream_is_not_ducked_when_only_its_name_identifies_it() {
            let streams = serde_json::json!([
                sink_input(
                    281,
                    "280",
                    serde_json::json!({
                        "application.name": "PipeWire ALSA [voxtype-onnx-migraphx]",
                        "node.name": "alsa_playback.voxtype-onnx-migraphx",
                        "media.name": "ALSA Playback"
                    })
                ),
                sink_input(
                    90,
                    "89",
                    serde_json::json!({
                        "application.name": "Firefox",
                        "application.process.binary": "firefox",
                        "application.process.id": "4000"
                    })
                ),
            ]);

            assert_eq!(ducked_indexes(&streams, &Value::Null, 1), vec![90]);
        }

        #[test]
        fn own_stream_is_recognised_by_the_process_id_of_its_client() {
            // Nothing on the stream names voxtype; only its client does.
            let streams = serde_json::json!([
                sink_input(7, "6", serde_json::json!({ "media.name": "Playback" })),
                sink_input(9, "8", serde_json::json!({ "media.name": "Playback" })),
            ]);
            let clients = serde_json::json!([
                { "index": 6, "properties": { "application.process.id": "4242" } },
                { "index": 8, "properties": { "application.process.id": "4243" } },
            ]);

            assert_eq!(ducked_indexes(&streams, &clients, 4242), vec![9]);
            assert_eq!(ducked_indexes(&streams, &clients, 4243), vec![7]);
            assert_eq!(ducked_indexes(&streams, &clients, 1), vec![7, 9]);
        }

        #[test]
        fn another_voxtype_process_is_recognised_by_its_client_binary() {
            let streams = serde_json::json!([sink_input(
                7,
                "6",
                serde_json::json!({ "media.name": "Playback" })
            )]);
            let clients = serde_json::json!([{
                "index": 6,
                "properties": {
                    "application.process.id": "777",
                    "application.process.binary": ".voxtype-wrapped"
                }
            }]);

            assert!(ducked_indexes(&streams, &clients, 1).is_empty());
        }

        #[test]
        fn other_programs_playing_through_alsa_are_still_ducked() {
            let streams = serde_json::json!([sink_input(
                12,
                "11",
                serde_json::json!({
                    "application.name": "PipeWire ALSA [mpv]",
                    "node.name": "alsa_playback.mpv"
                })
            )]);
            let clients = serde_json::json!([{
                "index": 11,
                "properties": {
                    "application.process.id": "500",
                    "application.process.binary": "mpv"
                }
            }]);

            assert_eq!(ducked_indexes(&streams, &clients, 1), vec![12]);
        }

        #[test]
        fn a_stream_with_no_client_or_properties_is_ducked() {
            let streams = serde_json::json!([{
                "index": 3,
                "client": "n/a",
                "mute": false,
                "corked": false,
                "volume": { "mono": { "value_percent": "100%" } }
            }]);

            assert_eq!(ducked_indexes(&streams, &Value::Null, 1), vec![3]);
        }

        #[test]
        fn scales_stream_volumes_relative_to_current_values() {
            // 70% of the amplitude = cube root of 0.7 on the cubic
            // percentage scale, applied per channel.
            assert_eq!(
                scaled_volumes(&["100%".to_string(), "80%".to_string()], 70),
                vec!["89%", "71%"]
            );
            assert_eq!(
                scaled_volumes(&["50%".to_string(), "25%".to_string()], 30),
                vec!["33%", "17%"]
            );
        }

        #[test]
        fn full_amplitude_leaves_volumes_unchanged() {
            assert_eq!(
                scaled_volumes(&["100%".to_string(), "37%".to_string()], 100),
                vec!["100%", "37%"]
            );
        }

        #[test]
        fn zero_amplitude_mutes() {
            assert_eq!(scaled_volumes(&["100%".to_string()], 0), vec!["0%"]);
        }

        #[test]
        fn half_amplitude_is_six_db_not_eighteen() {
            // 0.5^(1/3) = 0.7937: half the amplitude keeps ~79% of the cubic
            // percentage. The pre-change behaviour would have produced "50%",
            // which is only 12.5% of the amplitude.
            assert_eq!(scaled_volumes(&["100%".to_string()], 50), vec!["79%"]);
        }

        fn sink(name: &str, index: u32) -> Value {
            serde_json::json!({ "index": index, "name": name })
        }

        fn card(name: &str, profile: &str) -> Value {
            serde_json::json!({ "name": name, "active_profile": profile })
        }

        #[test]
        fn a_speaker_sink_is_ready_without_waiting_for_bluetooth() {
            let sinks = serde_json::json!([sink("alsa_output.pci.Speaker", 60)]);
            let route = audio_route("alsa_output.pci.Speaker", &sinks, &serde_json::json!([]));
            assert!(!route.bluetooth);
            assert!(route.ready);
            assert_eq!(route.sink_index, Some(60));
        }

        #[test]
        fn a_headset_profile_is_not_ready_to_resume() {
            let name = "bluez_output.7C_2A_CA_E8_36_C1.1";
            let route = audio_route(
                name,
                &serde_json::json!([sink(name, 10)]),
                &serde_json::json!([card("bluez_card.7C_2A_CA_E8_36_C1", "headset-head-unit")]),
            );
            assert!(route.bluetooth);
            assert!(!route.ready);
            assert_eq!(route.sink_index, None);
        }

        #[test]
        fn a2dp_uses_the_sink_that_came_back_for_the_same_address() {
            let sinks = serde_json::json!([
                sink("alsa_output.Speaker", 4),
                sink("bluez_output.7C_2A_CA_E8_36_C1.1", 108)
            ]);
            let cards = serde_json::json!([card("bluez_card.7C_2A_CA_E8_36_C1", "a2dp-sink")]);
            // The default sink name can still be the headset-profile node
            // while the A2DP node is the one that exists.
            let route = audio_route("bluez_output.7C_2A_CA_E8_36_C1.0", &sinks, &cards);
            assert!(route.ready);
            assert_eq!(route.sink_index, Some(108));
        }

        #[test]
        fn an_sbc_xq_profile_counts_as_a2dp() {
            let name = "bluez_output.AA_BB.1";
            let route = audio_route(
                name,
                &serde_json::json!([sink(name, 1)]),
                &serde_json::json!([card("bluez_card.AA_BB", "a2dp-sink-sbc_xq")]),
            );
            assert!(route.bluetooth);
            assert!(route.ready);
            assert_eq!(route.sink_index, Some(1));
        }

        #[test]
        fn a_foreign_stream_counts_and_voxtype_or_corked_streams_do_not() {
            let inputs = serde_json::json!([
                {"index": 1, "sink": 10, "mute": false, "corked": true, "properties": {}},
                {
                    "index": 2,
                    "sink": 10,
                    "mute": false,
                    "corked": false,
                    "properties": {"application.process.binary": "voxtype"}
                },
                {
                    "index": 3,
                    "sink": 11,
                    "mute": false,
                    "corked": false,
                    "properties": {"application.name": "Chromium"}
                },
                {
                    "index": 4,
                    "sink": 11,
                    "mute": true,
                    "corked": false,
                    "properties": {"application.name": "Chromium"}
                }
            ]);
            assert!(!sink_has_foreign_input(10, &inputs, &Value::Null, 1));
            assert!(sink_has_foreign_input(11, &inputs, &Value::Null, 1));
        }

        #[test]
        fn a_new_pause_cancels_the_resume_that_was_settling() {
            // Same capture resume_players does before it is spawned.
            let epoch = MEDIA_RESUME_EPOCH.load(Ordering::Relaxed);
            assert!(!media_resume_cancelled(epoch));
            let _ = bump_media_resume_epoch();
            assert!(media_resume_cancelled(epoch));
        }
    }
}

#[cfg(target_os = "linux")]
pub use imp::{duck_playing_audio, pause_playing_players, restore_ducked_audio, resume_players};

// On non-Linux targets MPRIS doesn't apply. Keep the public API stable
// so the daemon doesn't need to cfg-gate every call site.
#[cfg(not(target_os = "linux"))]
pub async fn pause_playing_players(_ignored: &[String]) -> Vec<String> {
    Vec::new()
}

#[cfg(not(target_os = "linux"))]
pub async fn resume_players(_players: Vec<String>) {}

#[cfg(not(target_os = "linux"))]
pub async fn duck_playing_audio(
    _volume_percent: u8,
    _fade_ms: u32,
) -> (Vec<DuckedMediaStream>, Option<tokio::task::JoinHandle<()>>) {
    (Vec::new(), None)
}

#[cfg(not(target_os = "linux"))]
pub async fn restore_ducked_audio(
    _streams: Vec<DuckedMediaStream>,
    _ducked_percent: u8,
    _fade_ms: u32,
) {
}
