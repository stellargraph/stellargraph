# Community detection on a terrorist attack data

This is an example of using Unsupervised GraphSAGE embeddings with the clustering to demonstrate how to solve community detection problem. The demo guides through the steps and shows the differences between "traditional" community detection (infomap) and the embeddings' approach.

## Requirements
This example assumes the `stellargraph` library and its requirements have been 
installed by following the installation instructions in the README 
of the library's [root directory](https://github.com/stellargraph/stellargraph).

## Data

The dataset used in this demo is available at https://www.kaggle.com/START-UMD/gtd. The Global Terrorism Database (GTD) is an open-source database including information on terrorist attacks around the world from 1970 through 2017. The GTD includes systematic data on domestic as well as international terrorist incidents that have occurred during this time period and now includes more than 180,000 attacks. The database is maintained by researchers at the National Consortium for the Study of Terrorism and Responses to Terrorism (START), headquartered at the University of Maryland. For information refer to the initial data source: https://www.start.umd.edu/gtd/.

To run the demo notebook, extract the data into a directory, and adjust the data path in the notebook pinting to the raw data

