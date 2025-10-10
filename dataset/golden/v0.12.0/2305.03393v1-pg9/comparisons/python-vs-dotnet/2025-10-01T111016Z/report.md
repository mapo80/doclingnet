# Python vs .NET Markdown Comparison

## Summary Metrics

| Metric | Python | .NET | Delta (.NET - Python) |
| --- | ---: | ---: | ---: |
| Lines | 21 | 39 | +18 |
| Words | 439 | 67 | -372 |
| Characters | 2829 | 335 | -2494 |
| Blank Lines | 7 | 2 | -5 |


## Similarity Scores

- SequenceMatcher ratio: **0.0164**
- Line-level Jaccard overlap: **0.0192** (1 shared / 52 total unique lines)


## Character Discrepancies

Characters missing from the .NET output compared to Python:
| Character | Count | ASCII |
| --- | ---: | ---: |
| ' ' | 626 | 32 |
| e | 205 | 101 |
| - | 128 | 45 |
| a | 119 | 97 |
| t | 108 | 116 |
| s | 97 | 115 |
| r | 89 | 114 |
| n | 88 | 110 |
| i | 81 | 105 |
| o | 71 | 111 |
| l | 66 | 108 |
| | | 62 | 124 |
| c | 55 | 99 |
| d | 53 | 100 |
| . | 48 | 46 |
| h | 45 | 104 |
| m | 42 | 109 |
| T | 39 | 84 |
| u | 36 | 117 |
| p | 34 | 112 |
| ... | ... | ... |

Spurious characters introduced by the .NET output:
| Character | Count | ASCII |
| --- | ---: | ---: |
| \n | 19 | 10 |
| & | 2 | 38 |
| } | 2 | 125 |
| { | 2 | 123 |
| C | 1 | 67 |
| _ | 1 | 95 |


## Line-Level Diff

```diff
--- python-cli/docling.md
+++ dotnet-cli/docling.md
@@ -1,21 +1,39 @@
-order to compute the TED score. Inference timing results for all experiments were obtained from the same machine on a single core with AMD EPYC 7763 CPU @2.45 GHz.
+re Recognition
+ts for &ll experiments
+AMD EPYC 7763
+ith
 
-## 5.1 Hyper Parameter Optimization
+ents
+7763
+des &
+le and
+} It is
+lightly
+C speed
+e sare
+122]. Ef-
+1aw that
+complex
+nterpart_
+erence
+2 (secs)
+2.73
+5.39
+1.97
 
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
+0.843 L
+938
+1.91
+915 { 0.859
+3.81
+931 } 0834
+1.22
+.942 | 0.857
+2
+1.931 { 0.824
+the best prediction
+roduced
+independently
+alone , then
+Dr] TabNet (395k
+Cal
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
      "lines": 39,
      "words": 67,
      "characters": 335,
      "blank_lines": 2
    }
  },
  "similarity": {
    "sequence_matcher_ratio": 0.01643489254108723,
    "line_jaccard": 0.019230769230769232,
    "shared_unique_lines": 1,
    "total_unique_lines": 52
  },
  "missing_characters": {
    "o": 71,
    "r": 89,
    "d": 53,
    "e": 205,
    " ": 626,
    "t": 108,
    "c": 55,
    "m": 42,
    "p": 34,
    "u": 36,
    "h": 45,
    "T": 39,
    "E": 7,
    "D": 6,
    "s": 97,
    ".": 48,
    "I": 4,
    "n": 88,
    "f": 26,
    "i": 81,
    "g": 20,
    "l": 66,
    "a": 119,
    "x": 3,
    "w": 11,
    "b": 32,
    "A": 7,
    "M": 13,
    "P": 16,
    "7": 6,
    "6": 5,
    "3": 6,
    "U": 1,
    "@": 1,
    "2": 11,
    "4": 10,
    "5": 15,
    "G": 1,
    "H": 15,
    "z": 3,
    "#": 9,
    "1": 6,
    "y": 17,
    "O": 14,
    "W": 2,
    "v": 16,
    "N": 5,
    ",": 11,
    "(": 8,
    ")": 9,
    "R": 1,
    "S": 11,
    "L": 22,
    "-": 128,
    "F": 4,
    "[": 2,
    "9": 22,
    "|": 62,
    "0": 33,
    "8": 7,
    "Q": 1,
    "k": 4,
    "q": 2,
    "=": 3,
    ":": 1
  },
  "spurious_characters": {
    "\n": 19,
    "&": 2,
    "C": 1,
    "}": 2,
    "_": 1,
    "{": 2
  }
}
```
