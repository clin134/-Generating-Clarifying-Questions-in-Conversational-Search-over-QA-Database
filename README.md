# -Generating-Clarifying-Questions-in-Conversational-Search-over-QA-Database
## construct dataset for annotation and run the IIR algorithm
``` 
  python clarifying.py -iir
```
## run the GPT3 model
```
 python use_gpt3.py
```
## run the T5 model
### Please use run_T5.ipynb
## run sentence similarity evaluation
```
  python clarifying.py -s
```
## process human evaluation results and perfrom ranking-based evaluation
```
  python claifying.py -r
```
