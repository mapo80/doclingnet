Optimized 'Lable 'LOkehizalioul Lor 'Lable Suructure Recoguitiol

[Page-header]

order to compute the LBD score: Intereuce timning results for all experiments were obtained Erom the same machiue ox & siugte core with AMD BPYC 7763 CPU @2 45 GHz

5.1 Hyper Parameter Optimization

We have chosen the Pub ZabNet data set to perform HPO, since it includes highly diverse set of tables: Also we report TED scores separately fOr Simple and complex tables (tables with cell spans). Results are presented in Table: 4 It is evident that with O2SL, Our model achieves the same TED score aud slightly better maP scorcs in comparison to HLML; However OLSL yields a %x speed up in the inference runtime over HTML;

Table 1. HPO performed 1n OTSL and HTML representation DII the Sallle transformer-based TableFormer [9] architecture; trained only on PubTabNet [22]: Ef- lects of reducing (he # 0l layers in encoder and decoder stages 01 the uodel show (hat Sualler models trained on OTSL perform better; especially in recoguizing complex lable struchures; Ald maittai a Iuch higher Ial scure Hhan Hhe HLLML couuterpan6

[Table]

5.2 Quantitative Results

We picked the model parameter confguration that produced the best prediction quality (enc-6, dec=6, heads=8) with PubZab Vet aloue; then independently trained and evaluated it on three _ publicly available data sets: PubTabNet (395k Salples); Lin ZabNet (LLSk sanpkes) and PubZables-IM (about IML samples)

We picked the model parameter confguration that produced the best prediction quality (enc-6, dec=6, heads=8) with PubZab Vet aloue; then independently trained and evaluated it on three publicly available data sets: PubTabNet (395k Samples) , Ein LabNet (113k samples ) and Pub Lables-IM (about IM samples) Performance results are presented in Table: p} It is clearly evident that the model trained on ( LSL outpertorms H LML across the board, kceping high 'IEDs and mAP scores even on difficult financial tables (Fin TabNet) that contain sparse and large tables.

Performauce results are presented in Iable: p} It is clearly evident that the model trained on OTSL outperforms HTML across the board, keeping high TEDs and maP scorcs even o=1 dificult financial tablcs (Lin LabNet) that coutain sparsc and large tables:

Additionally; the results show that OTSL has an advantage over  HTML when applied ox a bigger data set like PubZables-LM and achieves siguifcautly improved scorcs; Linaly; ( ISL achieves Easter interence due to tewer decoding steps which is a result of the reduced sequence representation;
