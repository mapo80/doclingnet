# Python vs .NET Markdown Comparison

## Summary Metrics

| Metric | Python | .NET | Delta (.NET - Python) |
| --- | ---: | ---: | ---: |
| Lines | 21 | 39 | +18 |
| Words | 439 | 424 | -15 |
| Characters | 2829 | 2417 | -412 |
| Blank Lines | 7 | 0 | -7 |


## Similarity Scores

- SequenceMatcher ratio: **0.2547**
- Line-level Jaccard overlap: **0.0000** (0 shared / 54 total unique lines)


## Character Discrepancies

Characters missing from the .NET output compared to Python:
| Character | Count | ASCII |
| --- | ---: | ---: |
| ' ' | 271 | 32 |
| - | 123 | 45 |
| . | 42 | 46 |
| e | 41 | 101 |
| n | 40 | 110 |
| T | 31 | 84 |
| s | 24 | 115 |
| f | 23 | 102 |
| m | 21 | 109 |
| t | 20 | 116 |
| o | 12 | 111 |
| r | 11 | 114 |
| i | 11 | 105 |
| P | 8 | 80 |
| 7 | 8 | 55 |
| | | 8 | 124 |
| # | 7 | 35 |
| 2 | 4 | 50 |
| d | 3 | 100 |
| D | 3 | 68 |
| ... | ... | ... |

Spurious characters introduced by the .NET output:
| Character | Count | ASCII |
| --- | ---: | ---: |
| L | 36 | 76 |
| c | 28 | 99 |
| , | 24 | 44 |
| I | 23 | 73 |
| 1 | 21 | 49 |
| \n | 19 | 10 |
| S | 17 | 83 |
| l | 16 | 108 |
| u | 15 | 117 |
| U | 14 | 85 |
| ; | 12 | 59 |
| "'" | 11 | 39 |
| O | 8 | 79 |
| H | 8 | 72 |
| k | 7 | 107 |
| x | 6 | 120 |
| Z | 6 | 90 |
| V | 6 | 86 |
| z | 5 | 122 |
| C | 5 | 67 |
| ... | ... | ... |


## Line-Level Diff

```diff
--- python-cli/docling.md
+++ dotnet-cli/docling.md
@@ -1,21 +1,39 @@
-order to compute the TED score. Inference timing results for all experiments were obtained from the same machine on a single core with AMD EPYC 7763 CPU @2.45 GHz.
-
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
+Optimized Table Tokenization for Table Structure Recognit
+order to coxpute Uhe 'IEXD) score, IuFerence 6iling results FO1' all experiments
+were Obtaized [ro11 the Sare machine O11 a Single core with AMD LlYC {63
+CPU @2.45 GHz.
+5.1 Hyper Parameter Optimization
+We have chosen the Pub ZabNet data set to perform HPO , since it includes a
+highly diversc sct oF tables, Also we repoxt 'IEL) scoros soparately Fi simple and
+complex tables (tables wIUh cell spans), Results are presented i 'Lable; || It is
+evident Uhak with OISL, Ouc modeL achieves the sane 'IED) score and slghely
+better Ial scores i1 coxparison1 to HZML; However (ZISL yields a Zc' speed
+up 1n the interence runtime over HTML
+Lable 1, HPO perlormed^ iu (LSL ald HLML representatiou 011 Uhe same
+translortler:-based 'LableFormter |V architecbure; trained only ou PulZabNet [224, EL:
+LecUs OL recHCHHG Vhe # OL HaverS L# CucOdeL Ald dccOder Suages O1 Vhe HdeL Show Vhal
+Sialel IOdek 6raled 01 (IS4 pelkoru bethel;, espocaHy I1 ICCOBHZLUL cUlplex
+kable sUrucUure8; Ald Iallai a Hch hgher IAL scure Uhan Uhe |LLML cutcrparg;,
+Language | TIEDs_Tx mAn | ,interence
+enc-layers | dec-layers | & zo | simple | complex | all | (U,/5) | Uime (secs)
+OTSL | 0.965 | 0.934 | 0.955 | 0.88 | 2.73
+HimL | 0,969 | 0,926 | 0.955 | 0,857 | 5,39
+01SL | 0,938 | 0,904 | 0,928 | 0,853 | 1,97
+HimL | 0,952 | U,9u9 | 0,938 | 0,843 | 3
+OISSL | 0,923 | 0.897 | 0,915 | 0,859 | 1,91
+HTML_| 0.945 L 0.901_|0.931 | 0.834 L_3.81
+OISSL | 0,952 | 0.92 |0,942 | 0,857 | 1,22
+HiML | 0,944 | 0.903 | 0,931 | 0.824 | '2
+5.2 Quantitative Results
+We picked the model parameter conliguration that produccd the best predictiol
+qualty (cuc-b; dcc=6, hcads=8) wIth !ub Lab et aloxe; Ghenl Iudopendeutly
+trained and evaluated I6 o1 thrce publcly available data scts; Lub Lab NVet (396k
+salpks| !I LahNct (LLSk SalpLcs| axd ! uu lablcs:- LML (ahuut LML Salplcs|
+P'ertormance results are presented in 'Iable |4 It is clearly evident that the model
+trained O11 (1SL outperkoris HLML across the board; kcopig hgh 1 EDs and
+AP scores evcn O11 dificult fnancial tables (VinTabNet) that contain spa
+and large tables
+Additioxally; the results Show that (15L has a11 advantage ovci' HiML
+when applicd 0u a bigger data Sct like Publables-IM and achieves siguifcantly
+Iiproved Scores, [ Izaly; (1SL achieves Laster inLerezce due to Lewer" docoding
+CpS WHcIl 1S & icsuLg OE the rccuccc Scqucncc rcprescngat1o
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
      "words": 424,
      "characters": 2417,
      "blank_lines": 0
    }
  },
  "similarity": {
    "sequence_matcher_ratio": 0.2546702249332825,
    "line_jaccard": 0.0,
    "shared_unique_lines": 0,
    "total_unique_lines": 54
  },
  "missing_characters": {
    "o": 12,
    "r": 11,
    "d": 3,
    "e": 41,
    " ": 271,
    "t": 20,
    "m": 21,
    "T": 31,
    "E": 2,
    "D": 3,
    "s": 24,
    ".": 42,
    "n": 40,
    "f": 23,
    "i": 11,
    "g": 1,
    "a": 3,
    "w": 1,
    "M": 2,
    "P": 8,
    "7": 8,
    "3": 1,
    "2": 4,
    "4": 1,
    "5": 1,
    "#": 7,
    "y": 3,
    "N": 1,
    ")": 2,
    "-": 123,
    "9": 1,
    "]": 2,
    "|": 8,
    "=": 1
  },
  "spurious_characters": {
    "O": 8,
    "p": 1,
    "z": 5,
    "l": 16,
    "k": 7,
    "S": 17,
    "u": 15,
    "c": 28,
    "R": 1,
    "\n": 19,
    "x": 6,
    "U": 14,
    "h": 1,
    "'": 11,
    "I": 23,
    "X": 1,
    ",": 24,
    "F": 1,
    "6": 2,
    "1": 21,
    "[": 1,
    "L": 36,
    "C": 5,
    "{": 1,
    "G": 2,
    "H": 8,
    "W": 1,
    "v": 1,
    "Z": 6,
    "(": 4,
    ";": 12,
    "^": 1,
    ":": 2,
    "V": 6,
    "B": 1,
    "8": 2,
    "_": 4,
    "&": 2,
    "/": 1,
    "!": 3,
    "\"": 1
  }
}
```