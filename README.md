# While way-finding, human is a *Genius* or a *Lazybone*?


### Abstract

Navigation in information networks is an essential part of our everyday lives. To design a user-friendly interactive system, it is significant to understand how humans build the link between two concepts. Different from how a computer does, humans will not always choose the shortest path to reach the destination. With the development of priori knowledge, people might have formed certain ways to finish the navigation. We dive into this topic with the help of *Wikispeedia* dataset. <br>
The goal of this project is to find the patterns that people tend to follow during the navigation paths. Is there an unrestrained creative navigation way or people tends to following a common pattern in the game?
There are two steps:
<ol>
<li> People establish logical chains, according to the priori knowledge.
<li> People change logical chains according to what they see on the real article pages.
</ol>
We will analyze two steps individually to answer this question.

### Research questions

<ol>
<li> Is there any specific pattern change in the articles' topic component, during the clicking process? For example, when navigating from ‘Zebra’ to ‘French Revolution’, there is a hypothesis that the elements of animals may decrease and the elements of history or politics may increase through the path. Does the ‘Zebra & French Revolution‘ hypothesis mentioned above or a similar phenomenon hold?
<li> Are there any other factors that influence people’s choice of clicking, i.e. contents on HTML pages? Do people tend to click links in an easy sentence or a professional sentence(readability)? Do people tend to click links that show up in the first few lines of the article(position)?
</ol>

### Methods

As we mention above, there are two main steps when people are trying to connect two different words:

<ol>
<li>People first establish a logical chain (clear or blurry) that connect the source word to the destination, preliminarily. This part is based on people's priori knowledge.</li>
<br>
<li>While scanning the HTML page, people change their initial logical chain and finish their "clicking task". People make their shift according to the HTML they see, i.e. plain texts and images in the page. </li>
</ol>

The construction of our methods is also based on those two main stages.

**Step 1**: Data scraping, pre-processing, and dataset construction. <br>

Import files in the dataset with pandas. Pre-processing the data stored in dataframes. Remove the unnecessary data point and handle the outlier and NaN.
Get familiar to the whole dataset from step 1.

**Step 2**: Basic Analysis <br>

Carry out the basic analysis and distribution plot on each dataframe. Validate the correctness of pre-processing. Create new columns which will be used in later in-depth research. For example, generate the variable "path_pair" from a path, "path_length" from a path, etc.

**Step 3**: In-depth Analysis <br>

Merge dataframes. Find the top highest-frequency path pattern that appeared in the dataset. Preliminarily analyze the relationship between those most common patterns and the data category, article, plain text.

**Step 4**: Topic modeling (LDA) <br>
Use the LDA model to analyze the first step when users are playing the game.
Utilize all the plain text of Wikipedia articles as the LDA model input to train an LDA model. Apply the LDA model to each article and output a topic distribution for each text. Within each article in one specific clicking path, we can see the shift of the topic distribution in each article component.
Steps in detail are as follows: <br>
<ol>
<li> LDA text preprocess
<li> Train the model and tune hyper-parameters with evaluation metric
<li> Label the final topics from the LDA model
<li> Run the LDA model on each article and construct a distribution plot of each article
<li> visualize the distribution shifting in the path
</ol>

**Step 5**: HTML analysis (Readability)

Analyze the top highest-frequency path pattern we find in step 3 with the HTML dataset.
<ol>
<li> Sample sentences that contain the specific path link in the most common path pattern. Construct a new dataset with those sentences selected. 
<li> Apply the readability analysis method to find whether people's tendency of clicking is related to the readability of those sentences. <br>
The metrics used to measure the readability include Flesch reading-ease, Flesch-Kincaid Grade Level, Dale Chall Readability, Automated Readability Index, Coleman Liau Index, and Gunning Fog. 
<li> Analyze the results from those evaluation metrics.
</ol>

**Step 6**: HTML analysis (Position)
Analyze the top highest-frequency path pattern we find in step 3 with the HTML dataset.
<ol>
<li> Cluster the links that appeared in each article's HTML page
<li> Analyse those clusters with the most common path pattern
</ol>


**Step 7**: HTML analysis (Image)
Analyse the top highest-frequency path pattern we find in step 3 with images in HTML pages.

**Step 8**: Github site-building and Datastory redaction.

Further details on each data pipeline processing step can be found in the notebook.

#### External libraries

### Proposed timeline
List all deadlines for every step above: <br>
28.10.22 **MileStone 2 release**
04.11.22 Data scraping, pre-processing, and dataset construction <br>
11.11.22 Basic Analysis; Topic modeling (LDA) <br>
17.11.22 In-depth Analysis; Topic modeling (LDA); HTML analysis (Readability); Integrate MileStone 2 <br>
18.11.22 **MileStone 2 deadline** <br>
25.11.22 Pause project work. Finish **Homework 2** <br>
02.12.22 Pause project work. Finish **Homework 2** <br>
09.12.22 HTML analysis (Readability); HTML analysis (Position) HTML analysis (Image) <br>
16.12.22 Complete all code implementations and visualisations relevant to analysis <br>
22.12.22 Github site-building and Datastory redaction； Integrate MileStone 3<br>
23.12.22 **MileStone 3 deadline** 

### Organization within the team

- @WayerLiu: LDA model; HTML analysis(Image); visualisations
- @Lyaoooo: in-depth analysis; HTML analysis(Position); visualisations
- @JackRuihang: in-depth analysis; HTML analysis(Readability); Github site
- @shcSteven: in-depth analysis; HTML analysis(Readability); Github site
