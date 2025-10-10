Optimized Table Tokenization for Table Structure Recognit
order to coxpute Uhe 'IEXD) score, IuFerence 6iling results FO1' all experiments
were Obtaized [ro11 the Sare machine O11 a Single core with AMD LlYC {63
CPU @2.45 GHz.
5.1 Hyper Parameter Optimization
We have chosen the Pub ZabNet data set to perform HPO , since it includes a
highly diversc sct oF tables, Also we repoxt 'IEL) scoros soparately Fi simple and
complex tables (tables wIUh cell spans), Results are presented i 'Lable; || It is
evident Uhak with OISL, Ouc modeL achieves the sane 'IED) score and slghely
better Ial scores i1 coxparison1 to HZML; However (ZISL yields a Zc' speed
up 1n the interence runtime over HTML
Lable 1, HPO perlormed^ iu (LSL ald HLML representatiou 011 Uhe same
translortler:-based 'LableFormter |V architecbure; trained only ou PulZabNet [224, EL:
LecUs OL recHCHHG Vhe # OL HaverS L# CucOdeL Ald dccOder Suages O1 Vhe HdeL Show Vhal
Sialel IOdek 6raled 01 (IS4 pelkoru bethel;, espocaHy I1 ICCOBHZLUL cUlplex
kable sUrucUure8; Ald Iallai a Hch hgher IAL scure Uhan Uhe |LLML cutcrparg;,
Language | TIEDs_Tx mAn | ,interence
enc-layers | dec-layers | & zo | simple | complex | all | (U,/5) | Uime (secs)
OTSL | 0.965 | 0.934 | 0.955 | 0.88 | 2.73
HimL | 0,969 | 0,926 | 0.955 | 0,857 | 5,39
01SL | 0,938 | 0,904 | 0,928 | 0,853 | 1,97
HimL | 0,952 | U,9u9 | 0,938 | 0,843 | 3
OISSL | 0,923 | 0.897 | 0,915 | 0,859 | 1,91
HTML_| 0.945 L 0.901_|0.931 | 0.834 L_3.81
OISSL | 0,952 | 0.92 |0,942 | 0,857 | 1,22
HiML | 0,944 | 0.903 | 0,931 | 0.824 | '2
5.2 Quantitative Results
We picked the model parameter conliguration that produccd the best predictiol
qualty (cuc-b; dcc=6, hcads=8) wIth !ub Lab et aloxe; Ghenl Iudopendeutly
trained and evaluated I6 o1 thrce publcly available data scts; Lub Lab NVet (396k
salpks| !I LahNct (LLSk SalpLcs| axd ! uu lablcs:- LML (ahuut LML Salplcs|
P'ertormance results are presented in 'Iable |4 It is clearly evident that the model
trained O11 (1SL outperkoris HLML across the board; kcopig hgh 1 EDs and
AP scores evcn O11 dificult fnancial tables (VinTabNet) that contain spa
and large tables
Additioxally; the results Show that (15L has a11 advantage ovci' HiML
when applicd 0u a bigger data Sct like Publables-IM and achieves siguifcantly
Iiproved Scores, [ Izaly; (1SL achieves Laster inLerezce due to Lewer" docoding
CpS WHcIl 1S & icsuLg OE the rccuccc Scqucncc rcprescngat1o
