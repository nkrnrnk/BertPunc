# BertPunc

This repository contains the code of BertPunc a punctuation restoration model based on Google's [BERT](https://arxiv.org/abs/1810.04805). The model is fine-tuned from a pretrained reimplementation of [BERT in Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT).

A punctation restoration model adds punctuation (e.g. period, comma, question mark) to an unsegmented, unpunctuated text. Automatic Speech Recognition (ASR) systems typically output unsegmented, unpunctuated sequences of words. Punctation restoration improves the readability of ASR transcripts.

## Results

BertPunc outperformes state-of-the-art results in [Bidirectional Recurrent Neural Network with Attention Mechanism for
Punctuation Restoration](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/1517.PDF) by Ottokar Tilk
and Tanel Alumae on the IWSLT dataset of Ted Talk transcripts by large margins:

| Model                    | Overall | Comma | Period | Question mark |
| ------------------------ |:-------:|:-----:|:------:|:-------------:|
| T-BRNN-pre (Tilk et al.) |  0.644  | 0.548 | 0.729  |        0.667  |
| BertPunc                 |  0.752  | 0.712 | 0.819  |        0.723  |
| *Improvement*            |  *+16%* | *+30%*| *+12%* |        *+8%*  |

(Scores are F1 scores on the test set)

## Method

BertPunc adds an extra linear layer on top of the pretrained BERT masked language model (BertForMaskedLM). BertForMaskedLM outputs a logit vector for every (masked) token. The logit vector has a 30522 size corresponding to the BERT token vocabulary. The extra linear layer maps to the possible punctutation characters (4 in case of the IWSL: comma, period, question mark and no punctuation).

BertPunc is trained by feeding it with word sequences of a fixed segment size. The label for a segment is the punctuation for the middle word in the sentence. The segment size is a hyperparameter, for the ISWLT dataset a size of 32 tokens works well.

## Code

* train.py: training code
* data.py: helper function to read and transform data
* model.py: neural network model
* evaluate.py: evaluation on ISWL test sets
