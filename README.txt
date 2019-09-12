12 September 2019

This file describes the data structure of the EU-IDA project folder.

Code/ contains the Python scripts
Data/ contains both the source and transformed data
Text/ contains the resulting figure files and Excel workbooks

Code/ in turn contains two subfolders: Code/Processing and Code/Analysis, the first generating the index decomposition analysis, and the second generating figures and tables. The sequence in which the scripts should be executed is:

Code/Processing/scrProcessInputs.py
Code/Processing/scrGenCoefficients.py
Code/Processing/scrCalcDecomposition.py

Code/Analysis/printBackground.py
Code/Analysis/plotAggregatePercent.py
Code/Analysis/perCapita.py
Code/Analysis/yearlyAcceleration.py
Code/Analysis/yearlyMemberStates.py
Code/Analysis/writeTables.py

Data/ in turn contains Data/Eurostat, which contains the original data, and Data/Processed, which contains the result of calculations.

Text/ contains the figures and tables with the corresponding data of the different parts of the manuscript: Text/Background, Text/Results and Text/Discussion, as well as a folder with the country-specific results, Text/Tables, and an Appendix with mathematical detail, Text/EU-electricity-IDA_Appendix.pdf  






