# Python vs .NET Markdown Comparison

## Summary Metrics

| Metric | Python | .NET | Delta (.NET - Python) |
| --- | ---: | ---: | ---: |
| Lines | 21 | 5 | -16 |
| Words | 439 | 393 | -46 |
| Characters | 2829 | 2512 | -317 |
| Blank Lines | 7 | 2 | -5 |


## Similarity Scores

- SequenceMatcher ratio: **0.0753**
- Line-level Jaccard overlap: **0.0556** (1 shared / 18 total unique lines)


## Character Discrepancies

Characters missing from the .NET output compared to Python:
| Character | Count | ASCII |
| --- | ---: | ---: |
| ' ' | 266 | 32 |
| - | 125 | 45 |
| . | 57 | 46 |
| | | 54 | 124 |
| 0 | 34 | 48 |
| 9 | 32 | 57 |
| T | 29 | 84 |
| 5 | 16 | 53 |
| 3 | 15 | 51 |
| 2 | 15 | 50 |
| \n | 15 | 10 |
| 4 | 13 | 52 |
| m | 11 | 109 |
| 7 | 11 | 55 |
| f | 10 | 102 |
| 8 | 10 | 56 |
| e | 9 | 101 |
| # | 8 | 35 |
| 1 | 3 | 49 |
| n | 2 | 110 |
| ... | ... | ... |

Spurious characters introduced by the .NET output:
| Character | Count | ASCII |
| --- | ---: | ---: |
| c | 58 | 99 |
| a | 37 | 97 |
| l | 35 | 108 |
| u | 26 | 117 |
| i | 22 | 105 |
| I | 22 | 73 |
| d | 20 | 100 |
| r | 17 | 114 |
| t | 16 | 116 |
| h | 16 | 104 |
| p | 15 | 112 |
| b | 14 | 98 |
| ; | 13 | 59 |
| Z | 12 | 90 |
| o | 11 | 111 |
| ( | 11 | 40 |
| k | 10 | 107 |
| S | 6 | 83 |
| N | 6 | 78 |
| z | 5 | 122 |
| ... | ... | ... |


## Line-Level Diff

```diff
--- python-cli/docling.md
+++ dotnet-cli/docling.md
@@ -1,21 +1,5 @@
-order to compute the TED score. Inference timing results for all experiments were obtained from the same machine on a single core with AMD EPYC 7763 CPU @2.45 GHz.
+Opbizized Table Tokenizabion Fol Table Structure Recognitiol Order to cxpute Hhe IHD score, Ihfcrcncc (izing resulks Fol All experimeuts were obtained {roxl Hhe Sanc machinc O11 a Sizgke core with AMD EPYC 7763 CPU @2.45 GHz 5.1 Hvper Parameter Optimization We have chosen Uhe PuN ZabNet data sct to perform HPO , Siuce it includcs & highly diverse set of tablcs, Also we repoxt 'IELD scores soparately For" simple and couplex tables (tables with cell spans) Results are presented in 'able: ||| It is cvident that with (ZXSL, Our" model achieves the sane 'IED) score and slightly better map scorcs il corparison to HZML; However (ISL yiclds a 2; speed up in the inference runtime over HTML Table 1, IIPO perlormed iu OTSL and FITML represcntatiox Ou1 Uhe Sahe translormer-based TableFormer |0 architecture; trained ouly on PubTabNet |22|, DL lects OF reducing Ahe # [ layers in elcoder and decoder slages OF Uhe model show (hal Bualler models trained On (ISL perlortl betker; especially in recoguizing complex table structures; and aintain a much Higher IAP score Hhan the IILML counterpart
 
-## 5.1 Hyper Parameter Optimization
+5.2 Quantitative Results We picked the model parameter coufguration that produccd the best predictiol quality (cnc=6, dcc=6, heads=8) with PublahNet alone; Hhen iudependently trained and evaluated it on threc publicly available data scts; LublabNet (805k Samples); EinZabNet (I18k samples) and PubZables-1M (about LML samples We picked the model parameter configuration that produced the best prcdictiol quality (cnc=6, dcc=6, heads=8) with PublahNet alone; Hhen iudependently trained and evaluatcd it o1 thrce publicly availablc data scts; PuLIabNct (BO6k Samples , Ein TahNct (LI3k sarplcs) and PuNZables-IM (aboug IML sampkes Performance rcsults arc presented in Iable | IF js clcarly cvidcnt (hak Uhe model trained on (ISL ouper forms HIML across Ahc board;, kceping High IEDs and IAp scorcs cvcll O11 cficult fnancial tables (Ein ZabNct) Uhat coutail sparsc and large tables.
 
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
+Performance results arc presented iu Iable | It is clcarly evident (hat (he mode trained on (ZSL outperforms HZML across the board, keeping high 'TEDs and mAp scores even1 011 difcult fnancial tables (LiwLabNct) that contaii1 Sparsc and large tables Additionally; the results show that OZSL has an advantage over HZML whel applicd 0x a bigger data Sct |ikc PubZables-LM and achieves siguifcantly improved scores, Einally; OISL achieves Easter inference due to fewer dccoding steps which is a result of the reduced scquencc represcntation
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
      "lines": 5,
      "words": 393,
      "characters": 2512,
      "blank_lines": 2
    }
  },
  "similarity": {
    "sequence_matcher_ratio": 0.07526680396929414,
    "line_jaccard": 0.05555555555555555,
    "shared_unique_lines": 1,
    "total_unique_lines": 18
  },
  "missing_characters": {
    "e": 9,
    " ": 266,
    "m": 11,
    "T": 29,
    "D": 1,
    ".": 57,
    "n": 2,
    "f": 10,
    "M": 1,
    "P": 1,
    "7": 11,
    "6": 1,
    "3": 15,
    "2": 15,
    "4": 13,
    "5": 16,
    "\n": 15,
    "#": 8,
    "1": 3,
    "y": 1,
    ")": 1,
    "-": 125,
    "[": 1,
    "9": 32,
    "]": 2,
    "|": 54,
    "0": 34,
    "8": 10
  },
  "spurious_characters": {
    "O": 2,
    "p": 15,
    "b": 14,
    "i": 22,
    "z": 5,
    "d": 20,
    "a": 37,
    "l": 35,
    "o": 11,
    "k": 10,
    "F": 4,
    "S": 6,
    "t": 16,
    "r": 17,
    "u": 26,
    "c": 58,
    "R": 1,
    "g": 2,
    "x": 3,
    "H": 2,
    "h": 16,
    "I": 22,
    "s": 4,
    ",": 2,
    "(": 11,
    "A": 1,
    "w": 2,
    "{": 1,
    "E": 1,
    "U": 5,
    "v": 5,
    "W": 1,
    "N": 6,
    "Z": 12,
    "&": 1,
    "'": 4,
    "L": 4,
    "\"": 2,
    "X": 1,
    ";": 13,
    "B": 2,
    "q": 1,
    "=": 3,
    "j": 1
  }
}
```