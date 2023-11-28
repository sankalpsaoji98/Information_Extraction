# Information_Extractor_Project

## Company and Position
Worked as a **Data Science Intern** at **Fidelity Investments**, a finance company based out of Boston MA, United States. This project is a small scale work motivated from the work we did there.

## Project
Goal of the project is to extract information via pattern matching with spaCy.

## Table of Contents
- [Software](#software)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Software
1) VSC Code
2) python3
3) spaCy

## Intro
When you go to the internet, one of the things that you want to check out is the current news. So, you go to many sites and check out many articles about a topic and get the information. But, imagine, just imagine, what if you get all the information extremely briefly and you just receive the particular sentences which show the extremely important information from the article like the location of the incident and the people involved. Won’t this be really time saving?

This is possible with Natural Language Processing (NLP). Wikipedia defines NLP as — “Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyse large amounts of natural language data”. In context to our problem, we can even extract information like location of an incident and people involved with NLP.

So, that’s what I have tried to do and this article shows how it is done. I have used Google Colab programming environment for the problem. ‘newspaper’ library has been used to get the data for the Natural Language Processing done with ‘spaCy’. Python is the base programming language used. Let’s get started now!

## Data
Now, we will go to a site and check for news. So, I came across a news related to protests in Myanmar. I will just copy the site address from the tab. The link is —

‘https://www.ndtv.com/world-news/11-dead-as-myanmar-protesters-fight-troops-with-handmade-guns-firebombs-2409266'

We will use this link for our project.

## Importing and Downloading the Required Libraries
For the project, in Google Colab, we will need to import,

```
import requests
import json
import pandas
import spacy
from spacy.matcher import Matcher
```
We will also need to import ‘newspaper’ library which is not pre-installed in Colab. So, we download it as,

```
!pip install newspaper3k
```

It will take a minute or so to get fully downloaded and installed. But, it will be temporary for that session of Colab. So, you got to run this command if you open a new session.

Now, we can do further imports,

```
import newspaper
from newspaper import Article
```

Now, since we imported spacy also, we will load the nlp module from it. It is done as below.

```
nlp = spacy.load('en_core_web_sm')
```

The ‘_sm’ part indicates it is a smaller version of the library. There are two more versions of the ‘en_core_web’ library. You can check those out!

## Getting the Data from the Site to our Notebook

It is time to define a function for getting the data from the news site to our Colab Notebook. We do it as,

```
def extract_article_from_url(url):
a = Article(url)
a.download()
a.parse()
return a.text, a.publish_date
```

We have used the Article method from the newspaper library. We are getting the text and the date of the news as output.

## Creating the Function for Sentence Output

Now, we will create the function that will give us the sentence output from our data for the matched word.

```
def sentence_giver(doc, start, end):
s = start
while str(doc[s]) != '.':
s = s - 1
if s < 0:
break
e = end
while str(doc[e]) != '.':
e = e + 1
if e > len(doc) - 1:
break
return str(doc[s+1:e])
```

Basically, what this function does is that it traverses till the nearest full stop in the article on both sides of the matched word. So, the indexes are reached both to the left and right of the matched word. Usually, we don’t see a full stop in the middle of the sentence a lot and mostly, it is at the ending of a sentence. Care is taken in case the matched word is in the first sentence or the last sentence of the article. So, we get proper sentences.

## Creating the Functions for Extracting Locations
Now, we are creating the function to extract the location of the incident. We do it as below.

```
def location_finder(doc):
locations = []
sentences = []
matcher = Matcher(nlp.vocab)
pattern = [{"ENT_TYPE": "GPE"}]
matcher.add("Location", [pattern])
matches = matcher(doc)
for match_id, start, end in matches:
sentence = sentence_giver(doc, start, end)
value = l_detail_filter(sentence)
if value == 1:
sentences.append(sentence)
locations.append(doc[start:end])
return locations, sentences
```

So, what we are doing is that we are using Matcher from spaCy to find the words from the text which come with entity type ‘GPE’ which applies for cities, countries, etc. Once we get such a word, we get its ID and with sentence giver, we get the sentence in which that word was found. Now, the function l_detail_filter checks if specific words from our patterns are in the sentence or not. If there are, we save our sentence and location. The function l_detail_filter is as below.

```
def l_detail_filter(data):
data = nlp(data)
matcher = Matcher(nlp.vocab)
pattern1 = [{"LOWER": "demonstrators"}]
matcher.add("L_Filter", [pattern1])
matches = matcher(data)
if len(matches) == 0:
return 0
else:
return 1
```

As we see, for now, I have set pattern1 as ‘demonstrators’. So, the idea is that if we get ‘demonstrators’ word in a sentence near a location, we can be sure that some people have gathered at that location and this means something is happening at that place! I have only added pattern1 but there can be more patterns added which can further help in selecting more specific sentences and locations.

## Creating the Functions for Extracting the Number of People Involved
So, now it’s time to get the number of people involved. The approach is very similar to what we did with locations. We first have the function for finding numbers.

```
def number_finder(doc):
numbers = []
sentences = []
matcher = Matcher(nlp.vocab)
pattern = [{"LIKE_NUM": True}]
matcher.add("Number", [pattern])
matches = matcher(doc)
for match_id, start, end in matches:
sentence = sentence_giver(doc, start, end)
value = n_detail_filter(sentence)
if value == 1:
sentences.append(sentence)
numbers.append(str(doc[start:end]))
return numbers, sentences
```

The IDs of the matched numbers are used to further get the sentences with the expected words with the below shown n_detail_filter function.

```
def n_detail_filter(data):
data = nlp(data)
matcher = Matcher(nlp.vocab)
pattern1 = [{"LOWER": "held"}, {"LOWER": "in"}, {"LOWER" : "detention"}]
matcher.add("N_Filter", [pattern1])
matches = matcher(data)
if len(matches) == 0:
return 0
else:
return 1
```

Thus, we get the number of people involved. I have just used the pattern as ‘held in detention’ but other patterns can also be used.

## Defining a Function to Get Results

Now, it’s time to integrate our above individual functions as a part of another function which can be used to get all the results together. We define a function named ‘solver’ to do it.

```
def solver(doc):
n_people, n_s = number_finder(doc)
print(n_people)
print(n_s)
l_places, l_s = location_finder(doc)
print(l_places)
print(l_s)
```

Now, we are ready to try this function on our data.

## Getting the Outputs
Now, we will try to get the outputs. First, we will get the article and date through the ‘extract_article_from_url’ function. Then, we will convert the news content to a NLP object.

```
link = 'https://www.ndtv.com/world-news/11-dead-as-myanmar-protesters-fight-troops-with-handmade-guns-firebombs-2409266'
content, date = extract_article_from_url(link)
doc = nlp(content)
```

Let’s print ‘doc’ to see how it looks.

```
print(doc)
```

We get,

This is how the article looks. Now, we get the results with the ‘solver’ function.

<img width="1385" alt="Screenshot 2023-11-28 at 11 56 30 AM" src="https://github.com/sankalpsaoji98/Information_Extraction/assets/26198596/cb2c8fdd-52ed-435e-87c4-7be6699eee57">

We get,

[‘2,847’]

[‘\n\nAAPP has said 2,847 people were currently being held in detention’]

[Myanmar, Myanmar]

[‘Anti-coup demonstrators in Myanmar fought back with handmade guns and firebombs\n\nAnti-coup demonstrators in Myanmar fought back with handmade guns and firebombs against a crackdown by security forces in a town in the northwest but at least 11 of the protesters were killed, domestic media reported on Thursday’, ‘Anti-coup demonstrators in Myanmar fought back with handmade guns and firebombs\n\nAnti-coup demonstrators in Myanmar fought back with handmade guns and firebombs against a crackdown by security forces in a town in the northwest but at least 11 of the protesters were killed, domestic media reported on Thursday’]

So, it works! We have got the number of people involved and also the location of the incident.

## Next Steps
I have just extracted ‘number of people’ and ‘location’ but other attributes could also be extracted. So, I hope this article has provided you the motivation to try out extracting other attributes from a news article. Keep going!


## Acknowledgements
The work was done for Sesto Synergy and is the property of the company.
