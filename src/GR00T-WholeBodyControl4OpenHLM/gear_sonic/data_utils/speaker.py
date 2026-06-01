"""
Speaker: TTS wrapper for Unitree G1 robot via unitree_sdk2_python AudioClient.

The robot's on-board Audio Service receives text over DDS (CycloneDDS) and
synthesises speech through its own TTS engine on the robot side.

Usage:
    from data_utils.speaker import Speaker

    speaker = Speaker(network_interface="eth0")
    speaker.speak("Get into teleop mode!")
"""

from __future__ import annotations

import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


# speaker_id used by TtsMaker:
#   0 → Chinese TTS engine on the robot
#   1 → English TTS engine on the robot
SPEAKER_ID_CHINESE = 0
SPEAKER_ID_ENGLISH = 1


class Speaker:
    """
    TTS wrapper that sends text to the Unitree G1 robot's Audio Service.

    TtsMaker is fire-and-forget (async on the robot side); the caller is
    responsible for sleeping long enough for playback to finish if needed.
    """

    def __init__(
        self,
        network_interface: str,
        volume: int = 85,
        speaker_id: int = SPEAKER_ID_ENGLISH,
        timeout: float = 10.0,
    ) -> None:
        """
        Args:
            network_interface: Network interface connected to the robot
                               (e.g. "eth0", "enP8p1s0").
            volume: Initial playback volume, 0-100.
            speaker_id: TTS engine on the robot (0=Chinese, 1=English).
            timeout: RPC call timeout in seconds.
        """
        # Initialize DDS channel factory before any SDK client is created
        ChannelFactoryInitialize(0, network_interface)

        self._client = AudioClient()
        self._client.SetTimeout(timeout)
        self._client.Init()
        self._client.SetVolume(volume)

        self._speaker_id = speaker_id

        # Fix upstream bug: tts_index starts at 0 and `+= self.tts_index`
        # keeps it at 0 forever.  We manage the counter ourselves.
        self._tts_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> int:
        """
        Send text to the robot's TTS engine (non-blocking on the caller side).

        Returns:
            0 on success, non-zero error code otherwise.
        """
        self._tts_index += 1
        # Patch the index directly so the SDK sends a fresh sequence number
        self._client.tts_index = self._tts_index
        return self._client.TtsMaker(text, self._speaker_id)

    def set_volume(self, volume: int) -> int:
        """
        Set playback volume.

        Args:
            volume: 0-100.

        Returns:
            0 on success, non-zero error code otherwise.
        """
        return self._client.SetVolume(volume)

    def get_volume(self) -> tuple[int, dict | None]:
        """
        Query current volume from the robot.

        Returns:
            (return_code, volume_dict) — volume_dict is None on failure.
        """
        return self._client.GetVolume()

    def set_led(self, r: int, g: int, b: int) -> int:
        """
        Control the RGB LED strip on the robot.

        Args:
            r, g, b: 0-255.
        """
        return self._client.LedControl(r, g, b)


# ---------------------------------------------------------------------------
# Quick smoke-test (requires a connected G1 robot)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface>")
        print("  e.g.: python3 speaker.py eth0")
        sys.exit(1)

    speaker = Speaker(network_interface=sys.argv[1], volume=85)

    speaker.speak("Get into teleop mode!")
    time.sleep(3)

    speaker.speak("Streaming data absent.")
    time.sleep(3)

    print("Done.")