use natural::phonetics::soundex;
use once_cell::sync::Lazy;
use regex::Regex;
use strsim::levenshtein;

/// Applies custom word corrections to transcribed text using fuzzy matching
///
/// This function corrects words in the input text by finding the best matches
/// from a list of custom words using a combination of:
/// - Levenshtein distance for string similarity
/// - Soundex phonetic matching for pronunciation similarity
///
/// # Arguments
/// * `text` - The input text to correct
/// * `custom_words` - List of custom words to match against
/// * `threshold` - Maximum similarity score to accept (0.0 = exact match, 1.0 = any match)
///
/// # Returns
/// The corrected text with custom words applied
pub fn apply_custom_words(text: &str, custom_words: &[String], threshold: f64) -> String {
    if custom_words.is_empty() {
        return text.to_string();
    }

    // Pre-compute lowercase versions to avoid repeated allocations
    let custom_words_lower: Vec<String> = custom_words.iter().map(|w| w.to_lowercase()).collect();

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut corrected_words = Vec::new();

    for word in words {
        let cleaned_word = word
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_lowercase();

        if cleaned_word.is_empty() {
            corrected_words.push(word.to_string());
            continue;
        }

        // Skip extremely long words to avoid performance issues
        if cleaned_word.len() > 50 {
            corrected_words.push(word.to_string());
            continue;
        }

        let mut best_match: Option<&String> = None;
        let mut best_score = f64::MAX;

        for (i, custom_word_lower) in custom_words_lower.iter().enumerate() {
            // Skip if lengths are too different (optimization)
            let len_diff = (cleaned_word.len() as i32 - custom_word_lower.len() as i32).abs();
            if len_diff > 5 {
                continue;
            }

            // Calculate Levenshtein distance (normalized by length)
            let levenshtein_dist = levenshtein(&cleaned_word, custom_word_lower);
            let max_len = cleaned_word.len().max(custom_word_lower.len()) as f64;
            let levenshtein_score = if max_len > 0.0 {
                levenshtein_dist as f64 / max_len
            } else {
                1.0
            };

            // Calculate phonetic similarity using Soundex
            let phonetic_match = soundex(&cleaned_word, custom_word_lower);

            // Combine scores: favor phonetic matches, but also consider string similarity
            let combined_score = if phonetic_match {
                levenshtein_score * 0.3 // Give significant boost to phonetic matches
            } else {
                levenshtein_score
            };

            // Accept if the score is good enough (configurable threshold)
            if combined_score < threshold && combined_score < best_score {
                best_match = Some(&custom_words[i]);
                best_score = combined_score;
            }
        }

        if let Some(replacement) = best_match {
            // Preserve the original case pattern as much as possible
            let corrected = preserve_case_pattern(word, replacement);

            // Preserve punctuation from original word
            let (prefix, suffix) = extract_punctuation(word);
            corrected_words.push(format!("{}{}{}", prefix, corrected, suffix));
        } else {
            corrected_words.push(word.to_string());
        }
    }

    corrected_words.join(" ")
}

/// Preserves the case pattern of the original word when applying a replacement
fn preserve_case_pattern(original: &str, replacement: &str) -> String {
    if original.chars().all(|c| c.is_uppercase()) {
        replacement.to_uppercase()
    } else if original.chars().next().map_or(false, |c| c.is_uppercase()) {
        let mut chars: Vec<char> = replacement.chars().collect();
        if let Some(first_char) = chars.get_mut(0) {
            *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
        }
        chars.into_iter().collect()
    } else {
        replacement.to_string()
    }
}

/// Extracts punctuation prefix and suffix from a word
fn extract_punctuation(word: &str) -> (&str, &str) {
    let prefix_end = word.chars().take_while(|c| !c.is_alphabetic()).count();
    let suffix_start = word
        .char_indices()
        .rev()
        .take_while(|(_, c)| !c.is_alphabetic())
        .count();

    let prefix = if prefix_end > 0 {
        &word[..prefix_end]
    } else {
        ""
    };

    let suffix = if suffix_start > 0 {
        &word[word.len() - suffix_start..]
    } else {
        ""
    };

    (prefix, suffix)
}

/// Filler words to remove from transcriptions
const FILLER_WORDS: &[&str] = &[
    "uh", "um", "uhm", "umm", "uhh", "uhhh", "ah", "eh", "hmm", "hm", "mmm", "mm", "mh", "ha",
    "ehh",
];

static MULTI_SPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

/// Patterns for repeated punctuation: "!!!!" -> "!" (regex crate doesn't support backrefs)
static REPEATED_EXCLAMATION: Lazy<Regex> = Lazy::new(|| Regex::new(r"!{3,}").unwrap());
static REPEATED_QUESTION: Lazy<Regex> = Lazy::new(|| Regex::new(r"\?{3,}").unwrap());
static REPEATED_PERIOD: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.{4,}").unwrap()); // Allow "..."
static REPEATED_COMMA: Lazy<Regex> = Lazy::new(|| Regex::new(r",{3,}").unwrap());
static REPEATED_DASH: Lazy<Regex> = Lazy::new(|| Regex::new(r"-{3,}").unwrap());

/// Removes repeated single characters like "f f f f f" (5+ occurrences)
/// This targets hallucination patterns, not stutters (which collapse_stutters handles at 3+)
fn remove_repeated_single_chars(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        // Check for single letter repeated with spaces (e.g., "f f f f f")
        if word_lower.len() == 1 && word_lower.chars().next().map_or(false, |c| c.is_alphabetic()) {
            let target_char = word_lower.chars().next().unwrap();
            let mut count = 1;
            while i + count < words.len() {
                let next_word = words[i + count].to_lowercase();
                if next_word.len() == 1 && next_word.chars().next() == Some(target_char) {
                    count += 1;
                } else {
                    break;
                }
            }
            // If 5+ repetitions, skip all of them (hallucination pattern)
            // 3-4 repetitions are handled by collapse_stutters as legitimate stutters
            if count >= 5 {
                i += count;
                continue;
            }
        }

        result.push(word);
        i += 1;
    }

    result.join(" ")
}

/// Common hallucination phrases that Whisper produces on silence/noise
const HALLUCINATION_PHRASES: &[&str] = &[
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitled by",
    "transcribed by",
    "[music]",
    "[applause]",
];

/// Detects GPU contention-induced hallucinations
///
/// When the GPU is under heavy load, Whisper can produce garbage output like:
/// - All punctuation: "!!!!!!!!", "...???.."
/// - Short text with excessive punctuation: "a!", "!a!"
/// - Repetitive single-word nonsense: "you you you you"
/// - Unicode garbage with non-printable characters
///
/// Returns true if the text appears to be GPU-induced garbage
pub fn is_gpu_contention_hallucination(text: &str) -> bool {
    let trimmed = text.trim();

    if trimmed.is_empty() {
        return true;
    }

    // Detect all-punctuation output (e.g., "!!!!!!!!", "...???..")
    let non_space_chars: Vec<char> = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    if !non_space_chars.is_empty() && non_space_chars.iter().all(|c| c.is_ascii_punctuation()) {
        return true;
    }

    // Detect short text with >=50% punctuation (e.g., "a!", "!a!")
    if trimmed.len() <= 5 && !non_space_chars.is_empty() {
        let punct_count = non_space_chars.iter().filter(|c| c.is_ascii_punctuation()).count();
        let punct_ratio = punct_count as f64 / non_space_chars.len() as f64;
        if punct_ratio >= 0.5 {
            return true;
        }
    }

    // Detect repetitive single-word nonsense (e.g., "you you you you")
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() >= 4 {
        let first_word = words[0].to_lowercase();
        let repetitive_count = words.iter().filter(|w| w.to_lowercase() == first_word).count();
        // If >75% of words are the same word, it's likely garbage
        if repetitive_count as f64 / words.len() as f64 > 0.75 {
            return true;
        }
    }

    // Detect unicode garbage (non-printable chars, excluding common whitespace)
    let has_garbage_chars = trimmed.chars().any(|c| {
        // Allow normal printable ASCII and common unicode letters/punctuation
        // Flag control characters, private use area, and other garbage
        c.is_control() && c != '\n' && c != '\r' && c != '\t'
            || ('\u{E000}'..='\u{F8FF}').contains(&c)  // Private Use Area
            || ('\u{FFF0}'..='\u{FFFF}').contains(&c)  // Specials block (replacement chars, etc.)
    });
    if has_garbage_chars {
        return true;
    }

    // Detect extremely short output with mostly non-alphabetic characters
    if trimmed.len() <= 3 {
        let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        if alpha_count == 0 {
            return true;
        }
    }

    false
}

/// Collapses repeated 1-2 letter words (3+ repetitions) to a single instance.
/// E.g., "wh wh wh wh" -> "wh", "I I I I" -> "I"
fn collapse_stutters(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        // Only process 1-2 letter words
        if word_lower.len() <= 2 && word_lower.chars().all(|c| c.is_alphabetic()) {
            // Count consecutive repetitions (case-insensitive)
            let mut count = 1;
            while i + count < words.len() && words[i + count].to_lowercase() == word_lower {
                count += 1;
            }

            // If 3+ repetitions, collapse to single instance
            if count >= 3 {
                result.push(word);
                i += count;
            } else {
                result.push(word);
                i += 1;
            }
        } else {
            result.push(word);
            i += 1;
        }
    }

    result.join(" ")
}

/// Pre-compiled filler word patterns (built lazily)
static FILLER_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    FILLER_WORDS
        .iter()
        .map(|word| {
            // Match filler word with word boundaries, optionally followed by comma or period
            Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))).unwrap()
        })
        .collect()
});

/// Detects if text is likely a Whisper hallucination (garbage output)
///
/// Returns true if the text appears to be:
/// - GPU contention hallucination (all punctuation, repetitive words, garbage chars)
/// - Empty or very short (fewer than 2 characters)
/// - Has extremely high ratio of non-alphabetic chars (>70%)
/// - Is extremely repetitive (10+ chars with only 1-2 unique characters)
/// - Contains known hallucination phrases
pub fn is_likely_hallucination(text: &str) -> bool {
    let trimmed = text.trim();

    // Check for GPU contention hallucinations first (fast check)
    if is_gpu_contention_hallucination(trimmed) {
        return true;
    }

    // Empty or very short text after trimming
    if trimmed.len() < 2 {
        return true;
    }

    // Check for known hallucination phrases (case-insensitive)
    let lower = trimmed.to_lowercase();
    for phrase in HALLUCINATION_PHRASES {
        if lower.contains(&phrase.to_lowercase()) {
            return true;
        }
    }

    // High ratio of non-alphabetic characters (>70%)
    let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    let total_non_space = trimmed.chars().filter(|c| !c.is_whitespace()).count();
    if total_non_space > 0 {
        let alpha_ratio = alpha_count as f64 / total_non_space as f64;
        if alpha_ratio < 0.3 {
            return true;
        }
    }

    // Extremely repetitive: 10+ chars with only 1-2 unique chars
    let non_space_chars: Vec<char> = trimmed
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if non_space_chars.len() >= 10 {
        let mut unique_chars: Vec<char> = non_space_chars.clone();
        unique_chars.sort();
        unique_chars.dedup();
        if unique_chars.len() <= 2 {
            return true;
        }
    }

    false
}

/// Filters transcription output by removing filler words, stutter artifacts, and hallucinations.
///
/// This function cleans up raw transcription text by:
/// 1. Collapsing repeated punctuation (e.g., "!!!" -> "!")
/// 2. Removing repeated single character patterns (e.g., "f f f f" -> "")
/// 3. Removing filler words (uh, um, hmm, etc.)
/// 4. Collapsing repeated 1-2 letter stutters (e.g., "wh wh wh" -> "wh")
/// 5. Cleaning up excess whitespace
///
/// # Arguments
/// * `text` - The raw transcription text to filter
///
/// # Returns
/// The filtered text with filler words, stutters, and hallucinations removed
pub fn filter_transcription_output(text: &str) -> String {
    let mut filtered = text.to_string();

    // Collapse repeated punctuation (e.g., "!!!" -> "!")
    filtered = REPEATED_EXCLAMATION.replace_all(&filtered, "!").to_string();
    filtered = REPEATED_QUESTION.replace_all(&filtered, "?").to_string();
    filtered = REPEATED_PERIOD.replace_all(&filtered, "...").to_string();
    filtered = REPEATED_COMMA.replace_all(&filtered, ",").to_string();
    filtered = REPEATED_DASH.replace_all(&filtered, "--").to_string();

    // Remove repeated single character patterns (e.g., "f f f f" -> "")
    filtered = remove_repeated_single_chars(&filtered);

    // Remove filler words
    for pattern in FILLER_PATTERNS.iter() {
        filtered = pattern.replace_all(&filtered, "").to_string();
    }

    // Collapse repeated 1-2 letter words (stutter artifacts like "wh wh wh wh")
    filtered = collapse_stutters(&filtered);

    // Clean up multiple spaces to single space
    filtered = MULTI_SPACE_PATTERN.replace_all(&filtered, " ").to_string();

    // Trim leading/trailing whitespace
    filtered.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_custom_words_exact_match() {
        let text = "hello world";
        let custom_words = vec!["Hello".to_string(), "World".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_apply_custom_words_fuzzy_match() {
        let text = "helo wrold";
        let custom_words = vec!["hello".to_string(), "world".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_preserve_case_pattern() {
        assert_eq!(preserve_case_pattern("HELLO", "world"), "WORLD");
        assert_eq!(preserve_case_pattern("Hello", "world"), "World");
        assert_eq!(preserve_case_pattern("hello", "WORLD"), "WORLD");
    }

    #[test]
    fn test_extract_punctuation() {
        assert_eq!(extract_punctuation("hello"), ("", ""));
        assert_eq!(extract_punctuation("!hello?"), ("!", "?"));
        assert_eq!(extract_punctuation("...hello..."), ("...", "..."));
    }

    #[test]
    fn test_empty_custom_words() {
        let text = "hello world";
        let custom_words = vec![];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_filter_filler_words() {
        let text = "So um I was thinking uh about this";
        let result = filter_transcription_output(text);
        assert_eq!(result, "So I was thinking about this");
    }

    #[test]
    fn test_filter_filler_words_case_insensitive() {
        let text = "UM this is UH a test";
        let result = filter_transcription_output(text);
        assert_eq!(result, "this is a test");
    }

    #[test]
    fn test_filter_filler_words_with_punctuation() {
        let text = "Well, um, I think, uh. that's right";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Well, I think, that's right");
    }

    #[test]
    fn test_filter_cleans_whitespace() {
        let text = "Hello    world   test";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Hello world test");
    }

    #[test]
    fn test_filter_trims() {
        let text = "  Hello world  ";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_filter_combined() {
        let text = "  Um, so I was, uh, thinking about this  ";
        let result = filter_transcription_output(text);
        assert_eq!(result, "so I was, thinking about this");
    }

    #[test]
    fn test_filter_preserves_valid_text() {
        let text = "This is a completely normal sentence.";
        let result = filter_transcription_output(text);
        assert_eq!(result, "This is a completely normal sentence.");
    }

    #[test]
    fn test_filter_stutter_collapse() {
        let text = "w wh wh wh wh wh wh wh wh wh why";
        let result = filter_transcription_output(text);
        assert_eq!(result, "w wh why");
    }

    #[test]
    fn test_filter_stutter_short_words() {
        let text = "I I I I think so so so so";
        let result = filter_transcription_output(text);
        assert_eq!(result, "I think so");
    }

    #[test]
    fn test_filter_stutter_mixed_case() {
        let text = "No NO no NO no";
        let result = filter_transcription_output(text);
        assert_eq!(result, "No");
    }

    #[test]
    fn test_filter_stutter_preserves_two_repetitions() {
        let text = "no no is fine";
        let result = filter_transcription_output(text);
        assert_eq!(result, "no no is fine");
    }

    #[test]
    fn test_filter_repeated_punctuation() {
        let text = "Hello!!! World???";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Hello! World?");
    }

    #[test]
    fn test_filter_repeated_chars() {
        // 5+ repetitions of single chars are removed as hallucinations
        let text = "f f f f f f hello";
        let result = filter_transcription_output(text);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_filter_preserves_normal_exclamation() {
        let text = "Really?! That's amazing!";
        let result = filter_transcription_output(text);
        assert_eq!(result, "Really?! That's amazing!");
    }

    #[test]
    fn test_is_hallucination_empty() {
        assert!(is_likely_hallucination(""));
        assert!(is_likely_hallucination("   "));
        assert!(is_likely_hallucination("a"));
    }

    #[test]
    fn test_is_hallucination_known_phrases() {
        assert!(is_likely_hallucination("Thank you for watching"));
        assert!(is_likely_hallucination("thanks for watching this video"));
        assert!(is_likely_hallucination("[music]"));
        assert!(is_likely_hallucination("[applause]"));
    }

    #[test]
    fn test_is_hallucination_non_alpha() {
        assert!(is_likely_hallucination("!!!!!!!!"));
        assert!(is_likely_hallucination("..........."));
        assert!(is_likely_hallucination("???!???!???"));
    }

    #[test]
    fn test_is_hallucination_repetitive() {
        assert!(is_likely_hallucination("ffffffffff"));
        assert!(is_likely_hallucination("aaaaaaaaaaaa"));
    }

    #[test]
    fn test_is_not_hallucination_valid() {
        assert!(!is_likely_hallucination("Hello world"));
        assert!(!is_likely_hallucination("This is a normal sentence."));
        assert!(!is_likely_hallucination("Testing 123"));
    }

    // GPU contention hallucination tests
    #[test]
    fn test_gpu_hallucination_all_punctuation() {
        // All-punctuation outputs are GPU contention hallucinations
        assert!(is_gpu_contention_hallucination("!!!!!!!!"));
        assert!(is_gpu_contention_hallucination("...???..."));
        assert!(is_gpu_contention_hallucination("!?!?!?!?"));
        assert!(is_gpu_contention_hallucination("---...---"));
        assert!(is_gpu_contention_hallucination(",,,"));
    }

    #[test]
    fn test_gpu_hallucination_short_with_punctuation() {
        // Short text with >50% punctuation is hallucination
        assert!(is_gpu_contention_hallucination("a!"));
        assert!(is_gpu_contention_hallucination("!a!"));
        assert!(is_gpu_contention_hallucination("..a"));
        // But normal short text is fine
        assert!(!is_gpu_contention_hallucination("hello"));
        assert!(!is_gpu_contention_hallucination("Hi"));
    }

    #[test]
    fn test_gpu_hallucination_repetitive_words() {
        // Repetitive single-word nonsense is hallucination
        assert!(is_gpu_contention_hallucination("you you you you"));
        assert!(is_gpu_contention_hallucination("the the the the the"));
        assert!(is_gpu_contention_hallucination("a a a a a a"));
        // But normal text with some repetition is fine
        assert!(!is_gpu_contention_hallucination("I think I think we should go"));
        assert!(!is_gpu_contention_hallucination("hello world"));
    }

    #[test]
    fn test_gpu_hallucination_unicode_garbage() {
        // Unicode garbage with control characters
        assert!(is_gpu_contention_hallucination("hello\x00world"));
        assert!(is_gpu_contention_hallucination("test\x1Fdata"));
        // Private use area characters
        assert!(is_gpu_contention_hallucination("text\u{E000}here"));
        // Normal unicode is fine
        assert!(!is_gpu_contention_hallucination("Hello café"));
        assert!(!is_gpu_contention_hallucination("日本語テスト"));
    }

    #[test]
    fn test_valid_transcriptions_not_flagged() {
        // Normal transcriptions should not be flagged
        assert!(!is_gpu_contention_hallucination("Hello, how are you today?"));
        assert!(!is_gpu_contention_hallucination("The quick brown fox jumps over the lazy dog."));
        assert!(!is_gpu_contention_hallucination("Testing 1 2 3"));
        assert!(!is_gpu_contention_hallucination("What's going on?"));
        assert!(!is_gpu_contention_hallucination("I can't believe it!"));
        // Short but valid
        assert!(!is_gpu_contention_hallucination("OK"));
        assert!(!is_gpu_contention_hallucination("yes"));
        assert!(!is_gpu_contention_hallucination("no"));
    }

    #[test]
    fn test_gpu_hallucination_empty_and_whitespace() {
        assert!(is_gpu_contention_hallucination(""));
        assert!(is_gpu_contention_hallucination("   "));
        assert!(is_gpu_contention_hallucination("\t\n"));
    }
}
