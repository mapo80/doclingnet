# Python vs .NET Markdown Comparison

## Summary Metrics

| Metric | Python | .NET | Delta (.NET - Python) |
| --- | ---: | ---: | ---: |
| Lines | 21 | 23 | +2 |
| Words | 439 | 70 | -369 |
| Characters | 2829 | 306 | -2523 |
| Blank Lines | 7 | 1 | -6 |


## Similarity Scores

- SequenceMatcher ratio: **0.0293**
- Line-level Jaccard overlap: **0.0270** (1 shared / 37 total unique lines)


## Character Discrepancies

Characters missing from the .NET output compared to Python:
| Character | Count | ASCII |
| --- | ---: | ---: |
| ' ' | 608 | 32 |
| e | 214 | 101 |
| - | 128 | 45 |
| a | 119 | 97 |
| t | 114 | 116 |
| s | 96 | 115 |
| r | 94 | 114 |
| n | 92 | 110 |
| i | 81 | 105 |
| o | 71 | 111 |
| l | 67 | 108 |
| c | 56 | 99 |
| d | 55 | 100 |
| | | 48 | 124 |
| h | 46 | 104 |
| . | 46 | 46 |
| m | 39 | 109 |
| T | 39 | 84 |
| p | 35 | 112 |
| u | 35 | 117 |
| ... | ... | ... |

Spurious characters introduced by the .NET output:
| Character | Count | ASCII |
| --- | ---: | ---: |
| \n | 3 | 10 |
| / | 3 | 47 |
| _ | 1 | 95 |


## Line-Level Diff

```diff
--- python-cli/docling.md
+++ dotnet-cli/docling.md
@@ -1,21 +1,23 @@
-order to compute the TED score. Inference timing results for all experiments were obtained from the same machine on a single core with AMD EPYC 7763 CPU @2.45 GHz.
+experiments
+EPYC 7763
+it includes a
+or simple and
+Table: /l/ It is
+and slightly
+ds a 21 speed
+on the same
+bNet |22/. Ef-
+)del show that
+izing complex
+L counterpart_
+Inference
+time (secs
+2.73
 
-## 5.1 Hyper Parameter Optimization
-
-We have chosen the PubTabNet data set to perform HPO, since it includes a highly diverse set of tables. Also we report TED scores separately for simple and complex tables (tables with cell spans). Results are presented in Table. 1. It is evident that with OTSL, our model achieves the same TED score and slightly better mAP scores in comparison to HTML. However OTSL yields a 2x speed up in the inference runtime over HTML.
-
-Table 1. HPO performed in OTSL and HTML representation on the same transformer-based TableFormer [9] architecture, trained only on PubTabNet [22]. Effects of reducing the # of layers in encoder and decoder stages of the model show that smaller models trained on OTSL perform better, especially in recognizing complex table structures, and maintain a much higher mAP score than the HTML counterpart.
-
-| # enc-layers   | # dec-layers   | Language   | TEDs        | TEDs        | TEDs        | mAP (0.75)   | Inference time (secs)   |
-|----------------|----------------|------------|-------------|-------------|-------------|--------------|-------------------------|
-| # enc-layers   | # dec-layers   | Language   | simple      | complex     | all         | mAP (0.75)   | Inference time (secs)   |
-| 6              | 6              | OTSL HTML  | 0.965 0.969 | 0.934 0.927 | 0.955 0.955 | 0.88 0.857   | 2.73 5.39               |
-| 4              | 4              | OTSL HTML  | 0.938 0.952 | 0.904 0.909 | 0.927 0.938 | 0.853 0.843  | 1.97 3.77               |
-| 2              | 4              | OTSL HTML  | 0.923 0.945 | 0.897 0.901 | 0.915 0.931 | 0.859 0.834  | 1.91 3.81               |
-| 4              | 2              | OTSL HTML  | 0.952 0.944 | 0.92 0.903  | 0.942 0.931 | 0.857 0.824  | 1.22 2                  |
-
-## 5.2 Quantitative Results
-
-We picked the model parameter configuration that produced the best prediction quality (enc=6, dec=6, heads=8) with PubTabNet alone, then independently trained and evaluated it on three publicly available data sets: PubTabNet (395k samples), FinTabNet (113k samples) and PubTables-1M (about 1M samples). Performance results are presented in Table. 2. It is clearly evident that the model trained on OTSL outperforms HTML across the board, keeping high TEDs and mAP scores even on di ffi cult financial tables (FinTabNet) that contain sparse and large tables.
-
-Additionally, the results show that OTSL has an advantage over HTML when applied on a bigger data set like PubTables-1M and achieves significantly improved scores. Finally, OTSL achieves faster inference due to fewer decoding steps which is a result of the reduced sequence representation.
+0 | U.oo (| 0.05
+7 | 0.853 | 1.97
+8 | 0.843 | 3.77
+5 | 0.859 | 1.91
+| 0.834 | 3.81
+2 | 0.857 | 1.22
+1 | 0.824 | 2
```

## Raw JSON Summary

```json
{
  "summary": {
    "python": {
      "lines": 21,
      "words": 439,
      "characters": 2829,
      "blank_lines": 7
    },
    "dotnet": {
      "lines": 23,
      "words": 70,
      "characters": 306,
      "blank_lines": 1
    }
  },
  "similarity": {
    "sequence_matcher_ratio": 0.02934609250398724,
    "line_jaccard": 0.02702702702702703,
    "shared_unique_lines": 1,
    "total_unique_lines": 37
  },
  "missing_characters": {
    "o": 71,
    "r": 94,
    "d": 55,
    "e": 214,
    " ": 608,
    "t": 114,
    "c": 56,
    "m": 39,
    "p": 35,
    "u": 35,
    "h": 46,
    "T": 39,
    "E": 7,
    "D": 8,
    "s": 96,
    ".": 46,
    "I": 3,
    "n": 92,
    "f": 26,
    "i": 81,
    "g": 20,
    "l": 67,
    "a": 119,
    "x": 3,
    "w": 11,
    "b": 32,
    "A": 8,
    "M": 14,
    "P": 16,
    "C": 1,
    "7": 5,
    "6": 6,
    "3": 10,
    "@": 1,
    "2": 11,
    "4": 11,
    "5": 15,
    "G": 1,
    "H": 15,
    "z": 2,
    "#": 9,
    "1": 10,
    "y": 18,
    "O": 14,
    "W": 2,
    "v": 16,
    "N": 5,
    ",": 12,
    "(": 8,
    ")": 9,
    "R": 2,
    "S": 11,
    "L": 22,
    "-": 128,
    "F": 4,
    "[": 2,
    "9": 29,
    "]": 2,
    "|": 48,
    "0": 29,
    "8": 6,
    "Q": 1,
    "k": 5,
    "q": 2,
    "=": 3
  },
  "spurious_characters": {
    "\n": 3,
    "/": 3,
    "_": 1
  }
}
```