# AICUP2020 醫病訊息去識別化
## Description
### About this compete
https://aidea-web.tw/aicup_meddialog  
This compete is to find out all the private information(e.g. 上午十點, 黃醫師, ...etc) from conversations of doctors and patients, and mark each private information token as a privacy type(e.g. name, money, time, ...etc).  
We entered at September and ended at the end of December.
### About our team
Three main members. All of us are university students. 
### About this project
Actually it's just a NER task. We have tried CRF method but it was not enough to handle this task. So finally we applied BiLSTM-CRF model with BERT embedding.  
Since tensorflow hub only publish BERT for chinese language, we chose a great third-party package [Kashgari](https://github.com/BrikerMan/Kashgari) to implement BERT embedding and RoBERTa embedding.

## References
1. [BERT](https://arxiv.org/abs/1810.04805)
2. [BiLSTM-CRF](https://arxiv.org/abs/1508.01991)