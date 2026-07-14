import inspect
import unittest
from unittest.mock import patch

from dramabox_api import DramaBoxTTSEngine


class PromptContractTests(unittest.TestCase):
    def test_build_prompt_keeps_description_outside_spoken_quotes(self):
        prompt = DramaBoxTTSEngine._build_prompt(
            "Arrival",
            "A calm, intimate narrator with restrained wonder",
        )
        self.assertEqual(
            prompt,
            'A calm, intimate narrator with restrained wonder speaks, "Arrival"',
        )

    def test_default_seed_requests_random_generation(self):
        signature = inspect.signature(DramaBoxTTSEngine.tts_inference)
        self.assertEqual(signature.parameters["seed"].default, -1)

    def test_description_fragment_gets_a_speaking_subject(self):
        normalized = DramaBoxTTSEngine._normalize_voice_description("warmly and with quiet wonder")
        self.assertEqual(normalized, "A narrator speaks warmly and with quiet wonder")

    def test_description_fragment_starting_with_preposition_is_attached_directly(self):
        normalized = DramaBoxTTSEngine._normalize_voice_description("with quiet wonder")
        self.assertEqual(normalized, "A narrator speaks with quiet wonder")

    def test_existing_speech_verb_is_preserved(self):
        normalized = DramaBoxTTSEngine._normalize_voice_description(
            "A calm audiobook narrator speaks with measured warmth"
        )
        self.assertEqual(normalized, "A calm audiobook narrator speaks with measured warmth")

    def test_participle_is_normalized_to_a_finite_speech_verb(self):
        normalized = DramaBoxTTSEngine._normalize_voice_description(
            "A relieved young woman speaking with bright excitement"
        )
        self.assertEqual(normalized, "A relieved young woman speaks with bright excitement")

    def test_random_seed_is_resolved_for_each_request(self):
        with patch("dramabox_api.tts_api.secrets.randbelow", side_effect=[123, 456]):
            self.assertEqual(DramaBoxTTSEngine._resolve_seed(-1), 123)
            self.assertEqual(DramaBoxTTSEngine._resolve_seed(-1), 456)

    def test_seed_rejects_values_below_random_sentinel(self):
        with self.assertRaisesRegex(ValueError, "-1"):
            DramaBoxTTSEngine._resolve_seed(-2)

    def test_speaker_description_without_verb_appends_speaks(self):
        normalized = DramaBoxTTSEngine._normalize_voice_description(
            "A calm, intimate narrator with restrained wonder"
        )
        self.assertEqual(normalized, "A calm, intimate narrator with restrained wonder speaks")


if __name__ == "__main__":
    unittest.main()
