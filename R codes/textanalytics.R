# Install required packages (if not installed)
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

packages <- c("googlesheets4", "tidyverse", "tm", "wordcloud", "RColorBrewer", 
              "syuzhet", "ggplot2", "textclean", "tidytext", "textstem", 
              "SnowballC", "topicmodels", "reshape2", "igraph", "ggraph", "widyr", "readxl", "udpipe")
lapply(packages, install_if_missing)

# Load libraries
library(googlesheets4)  # Google Sheets API access
library(tidyverse)      # Data manipulation and visualization (ggplot2, dplyr, etc.)
library(tm)             # Text mining functions (Corpus, Term-Document Matrix)
library(wordcloud)      # Word cloud generation
library(RColorBrewer)   # Color palettes for visualizations
library(syuzhet)        # Sentiment analysis tools
library(ggplot2)        # Advanced visualization
library(textclean)      # Cleaning and standardizing text
library(tidytext)       # Text mining with a tidy approach
library(textstem)       # Lemmatization of words
library(SnowballC)      # Stemming of words
library(topicmodels)    # Topic modeling (LDA, etc.)
library(reshape2)       # Data reshaping (for visualization compatibility)
library(igraph)         # Creates and analyzes network graphs (bigram/trigram networks)
library(ggraph)         # Extends ggplot2 for network visualization (works with igraph)
library(widyr)          # Calculates word co-occurrence in text
library(readxl)         # Reads Excel files
library(udpipe)         # NLP processing (POS tagging, lemmatization, Named Entity Recognition)

# Authenticate with Google Sheets (one-time, interactive login)
#gs4_auth()

# Read Google Sheets data (Replace with your actual sheet URL or Sheet ID)
#google_sheet_url <- ""
#data <- read_sheet(google_sheet_url, sheet = 1)  # Modify sheet index if needed

# CSV Data
setwd("/Users/jomarrabajante/Downloads/255 code")
data <- read_xlsx("survey_responses.xlsx", sheet = 1)

# Select the column containing qualitative responses 
colnames(data)[1] <- "Response" # Update column name
text_data <- data$Response  

# Drop NA values and convert to character
text_data <- na.omit(text_data) 
text_data <- as.character(text_data)

# Define custom stopwords before the pipeline
custom_stopwords <- setdiff(stopwords("en"), c("not", "no", "but"))  # Excluding important words

### TEXT CLEANING & PREPROCESSING ###
clean_text <- text_data %>%
  tolower() %>%
  removePunctuation() %>%
  removeNumbers() %>%
  removeWords(custom_stopwords) %>%  # Use custom stopwords here
  stripWhitespace()

# Convert to corpus for NLP processing
text_corpus <- Corpus(VectorSource(clean_text)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, custom_stopwords) %>%  # Use custom stopwords here
  tm_map(stripWhitespace)

# Apply Lemmatization (text normalization that reduces words to their base or root form (lemma), considering the context and morphology of the word)
text_corpus <- tm_map(text_corpus, content_transformer(lemmatize_strings))

### WORD CLOUD ###
tdm <- TermDocumentMatrix(text_corpus)
tdm_matrix <- as.matrix(tdm)
word_freq <- sort(rowSums(tdm_matrix), decreasing = TRUE)
word_freq_df <- data.frame(word = names(word_freq), freq = word_freq)

# Filter out very long words
word_freq_df <- word_freq_df[nchar(word_freq_df$word) < 20, ]

# Word Cloud Visualization
set.seed(123)
wordcloud(words = word_freq_df$word, freq = word_freq_df$freq, 
          min.freq = 2, max.words = 100,  
          scale = c(1.5, 0.3),  # Adjust font scaling
          colors = brewer.pal(min(length(word_freq_df$word), 8), "Dark2"))  # Ensure color palette matches word count

### BIGRAMS & TRIGRAMS ###
bigram_data <- tibble(text = clean_text) %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  count(bigram, sort = TRUE)

trigram_data <- tibble(text = clean_text) %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  count(trigram, sort = TRUE)

# Display top bigrams and trigrams
print(head(bigram_data, 10))
print(head(trigram_data, 10))

### SENTIMENT ANALYSIS ###
sentiments <- get_nrc_sentiment(clean_text) 
sentiment_scores <- colSums(sentiments)
sentiment_df <- data.frame(sentiment = names(sentiment_scores), score = sentiment_scores)

# Sentiment Analysis Bar Chart
ggplot(sentiment_df, aes(x = reorder(sentiment, -score), y = score, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Sentiment Analysis of Responses",
       x = "Sentiment Category",
       y = "Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

### EMOTION-SPECIFIC WORD CLOUDS ###

# Tokenize words from clean_text
tokenized_words <- unlist(strsplit(clean_text, " "))

# Compute sentiment for each word
word_sentiments <- get_nrc_sentiment(tokenized_words)

# Extract words with positive and negative sentiment
positive_words <- tokenized_words[word_sentiments$positive > 0]
negative_words <- tokenized_words[word_sentiments$negative > 0]

# Compute word frequencies for positive and negative words
positive_word_freq <- as.data.frame(table(positive_words))
negative_word_freq <- as.data.frame(table(negative_words))

# Rename columns
colnames(positive_word_freq) <- c("word", "freq")
colnames(negative_word_freq) <- c("word", "freq")

# Convert frequency to numeric
positive_word_freq$freq <- as.numeric(positive_word_freq$freq)
negative_word_freq$freq <- as.numeric(negative_word_freq$freq)

# Remove NA values
positive_word_freq <- na.omit(positive_word_freq)
negative_word_freq <- na.omit(negative_word_freq)

# Word cloud for positive words
if (nrow(positive_word_freq) > 0) {
  wordcloud(words = positive_word_freq$word, 
            freq = positive_word_freq$freq, 
            min.freq = 2, 
            max.words = 100, 
            scale = c(2.5, 0.5),  # Reduce max font size
            random.order = FALSE,  # Arrange by frequency
            rot.per = 0.2,  # Reduce rotated words
            colors = brewer.pal(8, "Greens"))
} else {
  print("No sufficient positive words for word cloud.")
}

# Word cloud for negative words
if (nrow(negative_word_freq) > 0) {
  wordcloud(words = negative_word_freq$word, 
            freq = negative_word_freq$freq, 
            min.freq = 2, 
            max.words = 100, 
            scale = c(2.5, 0.5),  # Reduce max font size
            random.order = FALSE,  # Arrange by frequency
            rot.per = 0.2,  # Reduce rotated words
            colors = brewer.pal(8, "Reds"))
} else {
  print("No sufficient negative words for word cloud.")
}

### TOPIC MODELING (LDA) ###

# Convert text corpus to Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(text_corpus)

# Check if DTM is valid (not empty)
if (nrow(dtm) > 0 && ncol(dtm) > 0) {
  
  # Remove overly sparse terms to improve topic extraction
  dtm <- removeSparseTerms(dtm, 0.99)  # Adjust sparsity level if needed
  
  # Ensure at least one non-zero row in DTM
  dtm_matrix <- as.matrix(dtm)
  dtm <- dtm[rowSums(dtm_matrix) > 0, ]
  
  # Final check: Proceed only if DTM still contains valid data
  if (nrow(dtm) > 0) {
    
    # Perform LDA with k=5 topics
    lda_model <- LDA(dtm, k = 5, control = list(seed = 1234))  
    
    # Extract top terms per topic
    topics <- tidy(lda_model, matrix = "beta")
    top_terms <- topics %>%
      group_by(topic) %>%
      top_n(10, beta) %>%
      ungroup() %>%
      arrange(topic, -beta)
    
    # Plot top terms per topic
    ggplot(top_terms, aes(x = reorder(term, -beta), y = beta, fill = factor(topic))) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~ topic, scales = "free") +
      theme_minimal() +
      labs(title = "Top Terms in Each Topic",
           x = "Term",
           y = "Probability") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
  } else {
    print("Warning: No valid terms left after sparsity filtering. Topic modeling skipped.")
  }
  
} else {
  print("Error: No valid words in Document-Term Matrix. Check text preprocessing.")
}

### NETWORK VISUALIZATION OF BIGRAMS ###
# Separate bigrams into two words
bigram_network <- bigram_data %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(n > 2) %>%
  graph_from_data_frame()

ggraph(bigram_network, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE) +
  theme_void() +
  labs(title = "Bigram Network")

### NETWORK VISUALIZATION OF TRIGRAMS ###
# Separate trigrams into three words
trigram_network <- trigram_data %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(n > 2) %>%
  graph_from_data_frame()

# Plot the trigram network
ggraph(trigram_network, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE) +
  theme_void() +
  labs(title = "Trigram Network")

### NAMED ENTITY RECOGNITION (NER) ###

# Load English Udpipe Model
ud_model <- "english-ewt-ud-2.5-191206.udpipe"
if (!file.exists(ud_model)) {
  udpipe_download_model(language = "english")
}
ud_model <- udpipe_load_model(ud_model)

ner_results <- udpipe_annotate(ud_model, x = text_data) %>%
  as.data.frame()

# Extract and Count Named Entities
ner_counts <- ner_results %>%
  filter(upos %in% c("PROPN", "NOUN")) %>%
  count(lemma, sort = TRUE)

# Print Named Entities
print(head(ner_counts, 10))

# Plot Named Entities
ggplot(ner_counts %>% top_n(10, n), aes(x = reorder(lemma, -n), y = n, fill = lemma)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top Named Entities", x = "Entity", y = "Count")

### Description of Algorithms ###
# 1. Text Preprocessing: Cleans raw text data by converting to lowercase, removing punctuation, numbers, stopwords, and extra whitespace.
# 2. Term Frequency Analysis: Counts word occurrences to identify the most frequently used terms.
# 3. Word Cloud Generation: Visualizes word frequency using a word cloud representation.
# 4. N-Gram Analysis: Extracts bigrams (two-word phrases) and trigrams (three-word phrases) to analyze common phrase patterns.
# 5. Sentiment Analysis: Uses NRC lexicon to classify text into various sentiments and emotions.
# 6. Topic Modeling (LDA): Extracts hidden topics from the text data using Latent Dirichlet Allocation (LDA).
# 7. Bigram/Trigram Network Analysis: Creates a network graph of common word pair/triple relationships to visualize word connections.
# 8. Named Entity Recognition (NER): Identifies and categorizes key entities in a text into predefined categories such as names of people, organizations, locations, dates, and more.