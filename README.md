# While way-finding, human is a *Genius* or a *Lazybone*?

### Data Story Website
Find our fantasic story [here](https://lyaoooo.github.io/adaproject.github.io/)!

### Abstract

Navigation in the information network is an essential part of our everyday lives. It is significant to understand how humans build the link between two words, in order to design user-friendly interactive system. It is a general idea that: different from how a computer process the task, humans will not always choose the shortest path to reach destination. And at the first glance, one might consider that different people will link two words in various way according to their prior knowledge. However, here we want to investigate whether there is a common and general pattern in human’s formation of the word navigating. During the way-finding procedure, do human generate fantastic ideas to finish the word to word linking? Or though people finish the task in various path, there still exist a potential pattern in human’s mind to link two topic words. We carry out our analysis by using the dataset collecting through the human-computation game Wikispeedia. <br> <br>
Our main approach consists in analyzing the finished path during the game and delving into the contents in HTML pages which people scan in their navigation actions. The analysis part could be divided into two primary sections: research on the potential pattern included in the finished path (internal factor) and study whether the contents in HTML pages will influence people’s way-finding between the source and destination (external factor).

### Research questions
In the story, we will focus on the following research questions:
<ol>
<li> Is there any specific pattern change in articles' topic component, during the clicking process? For example, when navigating from ‘Zebra’ to ‘French Revolution’, there is a hypothesis that the elements of animals may decrease, and the elements of history or politics would increase through the path. Does the ‘Zebra & French Revolution‘ hypothesis mentioned above or similar phenomenon hold?
<li> Are there any external factors that influence people’s choice of clicking, i.e. contents on HTML pages? 
<ol>
	- Do people tend to click links in an easy sentence or a professional sentence? <br>
	- Do people tend to click links that show up in the first a few lines of the article or links that have concentrated distribution in the webpage?
</ol>
</ol>

### Methods

As we mention above, there are two main steps when people are trying to connect two different words:

<ol>
<li>People first establish a logical chain (clear or blurry) which connect the source word to the destination, preliminarily. This part is based on people's priori knowledge.</li>
<br>
<li>While scanning the HTML page, people change their initial logical chain and finish their "clicking task". People make their shift according to the HTML they see, i.e. textplain, images in the page. </li>
</ol>

The construction of our methods is also based on those two main stages.

**Step 1**: Data scraping, pre-processing and dataset construction. <br>

Import files in the dataset with pandas. Pre-processing the data stored in dataframes. Remove the unneccesary data point and handle the outlier and NaN.
Get familiar to the whole dataset from step 1.

**Step 2**: Basic Analysis <br>

Carry out the basic analysis and distribution plot on each dataframe. Validate the correctness of pre-processing. Create new columns which will be used in later in-depth research. For example, generate the variable "path_pair" from path, "path_length" from path, etc.

**Step 3**: In-depth Analysis <br>

Merge dataframes together. Find the top highest-frequency path pattern appeared in the dataset. Preliminarily analyse the relationship between those most common pattern and the data category, article, plain-text.

**Step 4**: Topic modeling (LDA) <br>
Use the LDA model to analyse the first step when users are playing the game.
Utilize all the plain text of Wikipedia articles as the LDA model input to train a LDA model. Apply the LDA model to each article and output a topic distribution for each text. Within each article in one specific clicking path, we can clearly see the shift of the topic distribution in each article component.
Steps in detail are as following: <br>
<ol>
<li> LDA text preprocess
<li> Train the model and tune hyper-parameters with evaluation metric
<li> Label the final topics from LDA model
<li> Run LDA model on each article and construct a distribution plot of each article
<li> visualize the distribution shifting in the path
</ol>

**Step 5**: HTML analysis (Readability)

Analyse the top highest-frequency path pattern we find in step 3 with the HTML dataset.
<ol>
<li> Sample sentences that contain the specific path link in the most common path pattern. Construct new dataset with those sentences selected. 
<li> Apply readability analysis method to find whether people's tendency of clicking is related to the readability of those sentences. <br>
The metrics used to measure the readability including Flesch reading-ease, Flesch-Kincaid Grade Level, Dale Chall Readability, Automated Readability Index, Coleman Liau Index and Gunning Fog. 
<li> Analyse the results from those evaluation metrics.
</ol>

**Step 6**: HTML analysis (Position)
Analyse the top highest-frequency path pattern we find in step 3 with the HTML dataset.
<ol>
<li> Cluster the links appeared in each article HTML page
<li> Analyse those clusters with the most common path pattern
</ol>


**Step 7**: Github site building and Datastory redaction.

Further details on each data pipeline processing steps can be found in the notebook.

### Proposed timeline
List all deadlines for every step above: <br>
28.10.22 **MileStone 2 realese**
04.11.22 Data scraping, pre-processing and dataset construction <br>
11.11.22 Basic Analysis; Topic modeling (LDA) <br>
17.11.22 In-depth Analysis; Topic modeling (LDA); HTML analysis (Readability); Integrate MileStone 2 <br>
18.11.22 **MileStone 2 deadline** <br>
25.11.22 Pause project work. Finish **Homework 2** <br>
02.12.22 Pause project work. Finish **Homework 2** <br>
09.12.22 HTML analysis (Readability); HTML analysis (Position) <br>
16.12.22 Complete all code implementations and visualisations relevant to analysis <br>
22.12.22 Github site building and Datastory redaction； Integrate MileStone 3<br>
23.12.22 **MileStone 3 deadline** 

### Organization within the team

- @WayerLiu: LDA Model; In-depth Analysis; Visualisations
- @Lyaoooo: In-depth Analysis; Visualisations and Web Design
- @JackRuihang: In-depth Analysis; HTML Analysis(Readability and Position); Github Site; Write Web Content
- @shcSteven: In-depth Analysis; HTML Analysis(Readability and Position); Github Site; Write Web Content
