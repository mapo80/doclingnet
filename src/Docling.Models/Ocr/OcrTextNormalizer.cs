using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Docling.Models.Ocr;

/// <summary>
/// Provides utilities to normalise OCR text outputs so downstream builders receive clean tokens.
/// </summary>
public static class OcrTextNormalizer
{
    private static readonly Dictionary<char, string> LigatureMap = new()
    {
        ['ﬀ'] = "ff",
        ['ﬁ'] = "fi",
        ['ﬂ'] = "fl",
        ['ﬃ'] = "ffi",
        ['ﬄ'] = "ffl",
        ['æ'] = "ae",
        ['œ'] = "oe",
    };

    /// <summary>
    /// Normalises the supplied OCR text by removing control characters, collapsing whitespace and expanding ligatures.
    /// </summary>
    /// <param name="text">Raw text emitted by the OCR engine.</param>
    /// <returns>A trimmed string ready for paragraph assembly.</returns>
    public static string Normalize(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        // FormKC expands compatibility characters (including many ligatures) yet we still handle common ones explicitly
        // so that behaviour is deterministic across runtimes that may not ship the same Unicode tables.
        var normalized = text.Normalize(NormalizationForm.FormKD);
        var builder = new StringBuilder(normalized.Length);
        var lastWasWhitespace = false;

        foreach (var rune in normalized.EnumerateRunes())
        {
            if (Rune.IsWhiteSpace(rune))
            {
                if (!lastWasWhitespace)
                {
                    builder.Append(' ');
                    lastWasWhitespace = true;
                }

                continue;
            }

            var category = Rune.GetUnicodeCategory(rune);
            if (category == UnicodeCategory.Control)
            {
                if (!lastWasWhitespace)
                {
                    builder.Append(' ');
                    lastWasWhitespace = true;
                }

                continue;
            }

            if (category is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark)
            {
                continue;
            }

            lastWasWhitespace = false;

            if (LigatureMap.TryGetValue((char)rune.Value, out var mapped))
            {
                builder.Append(mapped);
                continue;
            }

            builder.Append(rune.ToString());
        }

        return builder.ToString().Trim();
    }
}
