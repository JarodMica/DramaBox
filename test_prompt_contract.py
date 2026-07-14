import unittest

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

    def test_speaker_description_without_verb_appends_speaks(self):
        normalized = DramaBoxTTSEngine._normalize_voice_description(
            "A calm, intimate narrator with restrained wonder"
        )
        self.assertEqual(normalized, "A calm, intimate narrator with restrained wonder speaks")


if __name__ == "__main__":
    unittest.main()
