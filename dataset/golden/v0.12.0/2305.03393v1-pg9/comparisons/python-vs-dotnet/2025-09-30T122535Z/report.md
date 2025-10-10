# Python vs .NET Markdown Comparison

## Summary Metrics

| Metric | Python | .NET | Delta (.NET - Python) |
| --- | ---: | ---: | ---: |
| Lines | 21 | 43 | +22 |
| Words | 439 | 389 | -50 |
| Characters | 2829 | 2310 | -519 |
| Blank Lines | 7 | 0 | -7 |


## Similarity Scores

- SequenceMatcher ratio: **0.1985**
- Line-level Jaccard overlap: **0.0000** (0 shared / 58 total unique lines)


## Character Discrepancies

Characters missing from the .NET output compared to Python:
| Character | Count | ASCII |
| --- | ---: | ---: |
| ' ' | 310 | 32 |
| - | 123 | 45 |
| . | 28 | 46 |
| n | 23 | 110 |
| | | 23 | 124 |
| s | 15 | 115 |
| r | 9 | 114 |
| , | 8 | 44 |
| o | 8 | 111 |
| i | 7 | 105 |
| m | 7 | 109 |
| # | 6 | 35 |
| 1 | 6 | 49 |
| 2 | 6 | 50 |
| 0 | 5 | 48 |
| 5 | 5 | 53 |
| l | 5 | 108 |
| ) | 4 | 41 |
| H | 4 | 72 |
| P | 4 | 80 |
| ... | ... | ... |

Spurious characters introduced by the .NET output:
| Character | Count | ASCII |
| --- | ---: | ---: |
| '\n' | 23 | 10 |
| e | 12 | 101 |
| b | 11 | 98 |
| _ | 8 | 95 |
| t | 7 | 116 |
| ; | 6 | 59 |
| I | 5 | 73 |
| O | 5 | 79 |
| " | 4 | 34 |
| & | 4 | 38 |
| } | 4 | 125 |
| f | 3 | 102 |
| v | 3 | 118 |
| % | 2 | 37 |
| / | 2 | 47 |
| 8 | 2 | 56 |
| R | 2 | 82 |
| c | 2 | 99 |
| g | 2 | 103 |
| z | 2 | 122 |
| ... | ... | ... |


## Line-Level Diff

```diff
--- python-cli/docling.md
+++ dotnet-cli/docling.md
@@ -1,21 +1,43 @@
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
+for Table Structure Recognit
+Optimized Table Tokenizatio
+ce timing resulte fot MD EPedze73
+compute tbe TED score: Inference single core with
+orcle obtained from the same machie
+CPTI @2.45 GHz
+1 Eyper Parameter Optimization to perform HPO, since it includend
+Ue have chosen the PubTabNet datanat TED scores separately for blehl It is
+We have veroe set of tables. Alsc we reporResults are presented in Table s
+highly diverbese(tables with cell spans) Regs the same TED score and gligteed
+complex tablesitha OTSL; Our model achieves {owever OTSL yields & ? speea
+evident that witres in comparison to HTML
+better mAP score8 runtime over HTMI
+up in the inference Lutation On the
+OTSL and HTML representPubTabNet |22}; Ef-
+Iable 1 HPQ perfeForener |9} architecture trainer sbnges of the model show that
+Iablforter-based Tablofoaye; 9} ercbder and decodeteciallys 0f tbeogaiaing complex
+fects of reducing tbe #On OTSL perform better; especthan the HTML counterpart
+smaller models traineaaintain & much higher IAP score
+table structures;, and maintam & 4TmAP T Iaterence
+TEblexT al | (0.75) (time (secs)
+# |ec-#yers | Language [ simple [complex | 0 955 | 0.8s / " 2.73
+c-layers ( dec-layers | oTsL | 0.965 | 0.934 | 0.955 | 0.857 |_6.39
+HTML | 0969 | 4927 | 0.927/ o.8s3 | " 1.97
+OTSL | 0.938 | 0.909 | 0.938 | 0843 |_ 377
+FTML | 0952 | 0%09 | @.915 | 0.859 | " 1.81
+OTSL | 0.923 | 0.897 | 0.931 | 0.834 | _ 8.81
+FTM_ | 0.945 | 09 40.942| 0.857 | " 1.22
+OTSL | 0.952 | 093 | 0.931 | D.824
+HTML_Lug44_L 0.903
+5.2 Quantitative Results caatation that produced the best preddenol
+We picked the model paracaetet Confith afThbNet aotetsheub dexeatdesgsk
+quality (enc_6, detedft B chree publicly avaiableidat Me(about IM samples)
+trained and evaluatet {Ll3k samples) and Pub ablearly evident tbat the model
+samples) , FinTabNet re presented in Table p} It is clead Ykeeping high TEDs and
+Performance cesults trerombtHTML Acroes the bEadNeepihat contain sparse
+ned O OTSL OutpeiffOulf financial tables
+mAP scores even 0L< mot has AD advantage over HTML
+and large bablly; the results show that Obable-JM and achieves signifcadtny
+Additilied on & bigger data set likeve_faster inference due to fewer decodug
+when applied e Finaly OTSL achieves fastee representation:
+improved scores refult % the reduced sequence
+steps which 18
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
      "lines": 43,
      "words": 389,
      "characters": 2310,
      "blank_lines": 0
    }
  },
  "similarity": {
    "sequence_matcher_ratio": 0.198482194979568,
    "line_jaccard": 0.0,
    "shared_unique_lines": 0,
    "total_unique_lines": 58
  },
  "missing_characters": {
    "o": 8,
    "r": 9,
    "d": 1,
    " ": 310,
    "m": 7,
    "h": 2,
    "T": 1,
    "D": 1,
    "s": 15,
    ".": 28,
    "n": 23,
    "i": 7,
    "l": 5,
    "a": 4,
    "w": 2,
    "P": 4,
    "7": 2,
    "6": 3,
    "3": 1,
    "2": 6,
    "4": 2,
    "5": 5,
    "H": 4,
    "#": 6,
    "1": 6,
    "y": 4,
    "N": 1,
    ",": 8,
    "(": 3,
    ")": 4,
    "S": 1,
    "-": 123,
    "9": 2,
    "]": 2,
    "|": 23,
    "0": 5,
    "=": 3
  },
  "spurious_characters": {
    "f": 3,
    "b": 11,
    "e": 12,
    "t": 7,
    "u": 1,
    "c": 2,
    "R": 2,
    "g": 2,
    "\n": 23,
    "O": 5,
    "z": 2,
    "k": 1,
    ":": 1,
    "I": 5,
    "@": 1,
    "v": 3,
    "L": 1,
    ";": 6,
    "{": 2,
    "&": 4,
    "?": 1,
    "8": 2,
    "}": 4,
    "Q": 1,
    "F": 1,
    "/": 2,
    "\"": 4,
    "_": 8,
    "%": 2,
    "B": 1,
    "<": 1,
    "J": 1
  }
}
```
