import Foundation

/// Few-shot prompts that drive the bidirectional KSL ↔ EN/SW translation
/// running on `LiteRTLMEngine`. The bundled `gemma-4-e2b-ksl.litertlm` was
/// fine-tuned for both directions, but the in-context examples here are what
/// give us consistent uppercase-gloss formatting on Path B and natural-sounding
/// English on Path A. Validated against the model with
/// `recognition/.../smoke_test_ksl.py` — keep these in sync if you tweak the
/// model or want to support more directions.
enum KSLPrompts {

    /// Path A: gloss tokens from the recognizer → fluent English sentence
    /// for AVSpeechSynthesizer to read.
    static let glossToEnglish = """
You are a Kenyan Sign Language interpreter. Translate the following KSL gloss into natural English. KSL gloss uses simple word order without verb tenses. Default to present tense unless context implies past or future.

Examples:

Gloss: ME PAIN HEAD
English: I have a headache.

Gloss: ME FIVE MONTH PREGNANT
English: I am five months pregnant.

Gloss: YESTERDAY ME GO MARKET
English: Yesterday I went to the market.

Gloss: HE NOT UNDERSTAND
English: He doesn't understand.

Gloss: YES HERE ME ID
English: Yes, here is my ID.

Now translate this gloss:

Gloss: {input}
English:
"""

    /// Path B: hearing-user English (from SFSpeechRecognizer) → KSL gloss
    /// list for the avatar renderer to play.
    static let englishToGloss = """
You are an expert Kenyan Sign Language interpreter. Convert English into KSL gloss.

CRITICAL RULES:
1. Output ONLY uppercase tokens, NO lowercase words
2. DROP all articles (a, an, the) and most prepositions (to, of, in, at, with, for)
3. DROP auxiliary verbs (is, are, was, were, will, would, do, does, did, have, has, been)
4. Use KSL word order: Topic-Comment, Time-Subject-Verb-Object
5. Question words (WHAT, WHERE, WHO, WHEN, HOW) go at the END
6. Possessives become "MY/YOUR/HIS/HER" before the noun
7. Use ME not I, YOU not you, HE/SHE not he/she

Examples (study the word order carefully):

English: I have a headache.
Gloss: ME PAIN HEAD

English: What is your name?
Gloss: YOU NAME WHAT

English: Where does it hurt?
Gloss: HURT WHERE

English: Yesterday I went to the market.
Gloss: YESTERDAY ME GO MARKET

English: How much do you want to send?
Gloss: YOU WANT SEND HOW MUCH

English: Have you given any medicine?
Gloss: YOU GIVE MEDICINE FINISH

English: What is wrong with your child?
Gloss: YOUR CHILD WRONG WHAT

English: Please show your ID.
Gloss: SHOW YOUR ID PLEASE

English: Don't worry, it will take only a moment.
Gloss: WORRY NOT MOMENT ONLY

English: I will prescribe a different medicine.
Gloss: ME PRESCRIBE DIFFERENT MEDICINE

Now convert this English (output ONLY uppercase KSL gloss, nothing else):

English: {input}
Gloss:
"""

    /// Render a prompt for the given task by substituting the user's input.
    /// `{input}` is the only placeholder; we just do plain string replacement
    /// rather than dragging in a templating dependency.
    static func render(task: GemmaTranslator.Task, input: String) -> String {
        let template: String
        switch task {
        case .kslToEnglish, .kslToSwahili:
            // SW path is currently routed through EN — the bundled model
            // wasn't fine-tuned on Swahili. Re-fine-tuning is a follow-up.
            template = glossToEnglish
        case .englishToKSL, .swahiliToKSL:
            template = englishToGloss
        }
        return template.replacingOccurrences(of: "{input}", with: input)
    }

    /// Sane decode budget per task. Glosses are short uppercase lists,
    /// sentences are longer. Keeping this tight matters for on-device
    /// latency — every extra token is ~50-100ms of decode on iPhone.
    static func maxTokens(for task: GemmaTranslator.Task) -> Int {
        switch task {
        case .englishToKSL, .swahiliToKSL: return 32
        case .kslToEnglish, .kslToSwahili: return 80
        }
    }
}
