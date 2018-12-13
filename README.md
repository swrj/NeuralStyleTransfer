# Instructions  
Download StyleTransfer.exe in /dist and follow instructions in app.
If Gooey is unable to install, follow these steps in StyleTransfer.py:  
* replace
```python
from gooey import Gooey, GooeyParser
```  
with  
```python
from argparse import ArgumentParser
```  
* remove @Gooey line  
* replace GooeyParser with ArgumentParser  
* remove every instance of widget and choices in the add_argument statements  
* run locally:
```python
python StyleTransfer.py
```  

## [Website with more information and results](https://swrj.github.io/NeuralStyleTransfer/)