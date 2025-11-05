
# Import necessary libraries for data processing
import random  # Used for random operations (though not actively used in this script)
import json    # Used to read and write JSON data
import os      # Used for operating system operations (though not actively used in this script)


# ============================================================================
# MAIN FUNCTION: Extract event labels and event relation labels from raw JSON
# ============================================================================
# This script processes the MAVEN-ERE (MAterials on eVents and ENtities for 
# Event Relation Extraction) dataset. It transforms the original JSON format 
# into a more structured format suitable for training event extraction models.
# ============================================================================

def extract_label(line_json, out_path):
    """
    Extract and structure event information from a single article in MAVEN-ERE dataset.
    
    This function processes one article from the dataset and creates a structured JSON
    containing:
    1. Article metadata (ID, title, text)
    2. Tokenized article with token positions
    3. Event mentions with their locations
    4. Event labels for each token (event vs. non-event)
    5. Event relation labels (coreference, temporal, causal, subevent)
    
    Args:
        line_json (dict): A dictionary containing the raw article data from MAVEN-ERE
        out_path (str): The file path where the processed JSON will be saved
    
    Returns:
        None (writes output to file)
    """

    # ========================================================================
    # STEP 1: Extract basic article metadata
    # ========================================================================
    # Initialize the output dictionary that will contain all processed data
    article_json = {}
    
    # Store the unique article identifier from the original dataset
    article_json['article_id'] = line_json['id']
    
    # Store the article title
    article_json['article_title'] = line_json['title']
    
    # Combine all sentences into a single article text string
    # The original data has sentences as separate items in a list
    article_json['article_txt'] = " ".join(line_json['sentences'])

    # ========================================================================
    # STEP 2: Build a flattened tokens list from all sentences
    # ========================================================================
    # The original data has tokens organized by sentence. We need to flatten
    # this into a single list while keeping track of sentence boundaries.
    # This makes it easier to map event mentions to token positions later.
    
    # Initialize list to hold all tokens from all sentences in order
    article_json['tokens_list'] = []
    
    # Track how many tokens are in each sentence (for reference)
    num_tokens_in_sentence = []
    
    # Create a prefix sum array to quickly find the starting position 
    # of tokens in each sentence within the flattened tokens_list
    # prefix_sum_num_tokens[i] = total tokens before sentence i
    prefix_sum_num_tokens = [0]
    sum = 0

    # Iterate through each sentence in the article
    for sent_i in range(len(line_json['tokens'])):
        # Record the number of tokens in this sentence
        num_tokens_in_sentence.append(len(line_json['tokens'][sent_i]))
        
        # Update the running sum of total tokens seen so far
        sum += len(line_json['tokens'][sent_i])
        
        # Store the cumulative count (position where next sentence starts)
        prefix_sum_num_tokens.append(sum)
        
        # Add all tokens from this sentence to our flattened list
        article_json['tokens_list'].extend(line_json['tokens'][sent_i])


    # ========================================================================
    # STEP 3: Extract event mentions and map them to token positions
    # ========================================================================
    # Events in MAVEN-ERE can have multiple mentions (same event referred to 
    # in different parts of the text). We need to extract each mention and
    # determine its exact position in our flattened tokens_list.
    
    # Initialize list to store all event mention dictionaries
    # Each dict contains: event_id, mention_id, trigger_word, 
    # index_in_tokens_list, and later index_in_event_label
    article_json['event_mentions'] = []
    
    # Iterate through all events in the article
    for event_i in range(len(line_json['events'])):
        # Each event can have multiple mentions (different references to same event)
        for mention_i in range(len(line_json['events'][event_i]['mention'])):
            # Create a dictionary for this specific event mention
            event_dict = {}
            
            # Store the unique event ID (multiple mentions share same event_id)
            event_dict['event_id'] = line_json['events'][event_i]['id']
            
            # Store the unique mention ID
            event_dict['mention_id'] = line_json['events'][event_i]['mention'][mention_i]['id']
            
            # Store the trigger word (the word/phrase that indicates the event)
            event_dict['trigger_word'] = line_json['events'][event_i]['mention'][mention_i]['trigger_word']
            
            # Calculate the position of this event mention in the flattened tokens_list
            # The original data gives us:
            # - sent_id: which sentence the mention is in
            # - offset: [start, end) token indices within that sentence
            index_in_tokens_list = []
            
            # Get the sentence ID where this mention appears
            sent_id = line_json['events'][event_i]['mention'][mention_i]['sent_id']
            
            # Get the token offset range [start, end) within that sentence
            offset_start = line_json['events'][event_i]['mention'][mention_i]['offset'][0]
            offset_end = line_json['events'][event_i]['mention'][mention_i]['offset'][1]
            
            # Convert sentence-relative positions to absolute positions in tokens_list
            # prefix_sum_num_tokens[sent_id] gives us where the sentence starts
            # Then we add the offset within that sentence
            for token_i in range(offset_start, offset_end):
                index_in_tokens_list.append(prefix_sum_num_tokens[sent_id] + token_i)

            # ====================================================================
            # Validation: Verify that the tokens at these positions match the 
            # trigger_word from the original data
            # ====================================================================
            trigger_word = []
            for i in range(len(index_in_tokens_list)):
                trigger_word.append(article_json['tokens_list'][index_in_tokens_list[i]])
            
            # If they don't match, something went wrong with our position mapping
            if " ".join(trigger_word) != event_dict['trigger_word']:
                print("event trigger word not match")

            # Store the calculated positions
            event_dict['index_in_tokens_list'] = index_in_tokens_list
            
            # Add this event mention to our list
            article_json['event_mentions'].append(event_dict)

    # ========================================================================
    # Sort event mentions by their position in the text (natural reading order)
    # ========================================================================
    # This ensures events appear in the order they occur in the article,
    # making it easier to process sequential relationships
    article_json['event_mentions'] = sorted(article_json['event_mentions'], 
                                           key = lambda d: d['index_in_tokens_list'][0])


    # ========================================================================
    # STEP 4: Create event labels for every token in the article
    # ========================================================================
    # We need to create a sequence of labels where each element corresponds
    # to either a single token (non-event) or a multi-token event trigger.
    # This is useful for training sequence labeling models.
    #
    # Each element in event_label has:
    # - token: the actual word(s)
    # - index_in_tokens_list: position(s) in the flattened token list
    # - event_label: 0 for non-event, 1 for event
    # - index_in_event_label: position in this event_label list
    
    article_json['event_label'] = []
    
    # Use three pointers to traverse and merge tokens and events:
    point_token = 0        # Points to current position in tokens_list
    point_event = 0        # Points to current event in event_mentions
    point_event_label = 0  # Points to current position in event_label (output)

    # Continue until we've processed all tokens
    while point_token < len(article_json['tokens_list']) and point_event <= len(article_json['event_mentions']):
        
        # Check if we still have events to process
        if point_event < len(article_json['event_mentions']):
            
            # CASE 1: Current token is BEFORE the next event mention
            # This is a regular non-event token
            if point_token < article_json['event_mentions'][point_event]['index_in_tokens_list'][0]:
                token_dict = {}
                # Single token (not part of an event)
                token_dict['token'] = article_json['tokens_list'][point_token]
                # Wrap in list for consistency with multi-token events
                token_dict['index_in_tokens_list'] = [point_token]
                # Label as non-event
                token_dict['event_label'] = 0
                # Track position in event_label sequence
                token_dict['index_in_event_label'] = point_event_label
                
                # Add to event_label list
                article_json['event_label'].append(token_dict)
                
                # Move to next token
                point_token += 1
                point_event_label += 1
                
            # CASE 2: Current token position matches an event mention
            # This is an event trigger (possibly multi-token)
            else:
                token_dict = {}
                # Use the full trigger phrase (may be multiple tokens)
                token_dict['token'] = article_json['event_mentions'][point_event]['trigger_word']
                # Store all token positions that make up this event
                token_dict['index_in_tokens_list'] = article_json['event_mentions'][point_event]['index_in_tokens_list']
                # Label as event
                token_dict['event_label'] = 1
                # Track position in event_label sequence
                token_dict['index_in_event_label'] = point_event_label
                
                # Add to event_label list
                article_json['event_label'].append(token_dict)
                
                # Back-reference: store where this event appears in event_label
                article_json['event_mentions'][point_event]['index_in_event_label'] = point_event_label
                
                # Skip past all tokens in this event trigger
                point_token = article_json['event_mentions'][point_event]['index_in_tokens_list'][-1] + 1
                # Move to next event
                point_event += 1
                point_event_label += 1
                
        # CASE 3: All events have been processed, but tokens remain
        # These are non-event tokens at the end of the article
        else:
            token_dict = {}
            token_dict['token'] = article_json['tokens_list'][point_token]
            token_dict['index_in_tokens_list'] = [point_token]
            token_dict['event_label'] = 0  # Non-event
            token_dict['index_in_event_label'] = point_event_label
            
            article_json['event_label'].append(token_dict)
            
            point_token += 1
            point_event_label += 1



    # ========================================================================
    # STEP 5: Initialize event relation labels for all event pairs
    # ========================================================================
    # We need to create labels for every possible pair of events in the article.
    # Each pair will have 4 relation types:
    # 1. Coreference: whether two mentions refer to the same event
    # 2. Temporal: before(1), after(2), overlap(3), or none(0)
    # 3. Causal: causes(1), caused_by(2), or none(0)
    # 4. Subevent: contains(1), contained_by(2), or none(0)
    #
    # We only create pairs where i < j (ordered pairs, not all permutations)
    
    article_json['relation_label'] = []
    num_events = len(article_json['event_mentions'])  # Total number of event mentions

    # Create all possible pairs (i, j) where i < j
    for i in range(0, num_events - 1):
        for j in range(i + 1, num_events):
            event_pair = {}
            
            # Store references to both events in this pair
            event_pair['event_1'] = article_json['event_mentions'][i]
            event_pair['event_2'] = article_json['event_mentions'][j]

            # Initialize all relation labels to 0 (no relation by default)
            event_pair['label_coreference'] = 0
            event_pair['label_temporal'] = 0
            event_pair['label_causal'] = 0
            event_pair['label_subevent'] = 0

            # Check for coreference: if both mentions share the same event_id,
            # they are different mentions of the same event (coreferent)
            if event_pair['event_1']['event_id'] == event_pair['event_2']['event_id']:
                event_pair['label_coreference'] = 1

            # Add this pair to our relation_label list
            article_json['relation_label'].append(event_pair)

    # ====================================================================
    # Validation: Check that we created the correct number of pairs
    # ====================================================================
    # For n events, we should have n*(n-1)/2 pairs (combination formula)
    # This is equivalent to: (1 + (n-1)) * (n-1) / 2
    if len(article_json['relation_label']) != (1 + num_events - 1) * (num_events - 1) // 2:
        print("wrong number of event pairs")


    # ========================================================================
    # STEP 6: Process TEMPORAL relations - BEFORE type
    # ========================================================================
    # Temporal relations describe when events occur relative to each other.
    # BEFORE relation: event A happens before event B in time
    #
    # Label encoding:
    # 0 = no temporal relation
    # 1 = event_1 happens before event_2
    # 2 = event_1 happens after event_2
    # 3 = event_1 overlaps with event_2
    
    for relation_i in range(len(line_json['temporal_relations']['BEFORE'])):
        # Get the event IDs involved in this BEFORE relation
        relation_1 = line_json['temporal_relations']['BEFORE'][relation_i][0]
        relation_2 = line_json['temporal_relations']['BEFORE'][relation_i][1]

        # Filter: Only process relations between EVENTs (not TIME expressions)
        # MAVEN-ERE also includes temporal relations with time expressions,
        # but we only care about event-event relations
        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            # Find all mentions of relation_1 event
            relation_1_event = []
            relation_1_index_in_event_mentions = []
            # Find all mentions of relation_2 event
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            
            # Search through all event mentions to find those matching our event IDs
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            # Apply the BEFORE relation to all pairs of mentions
            # (If event A has 2 mentions and event B has 3 mentions,
            #  we create 2*3=6 pairwise relations)
            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    
                    # We need to find this pair in our relation_label list
                    # The pairs are ordered by textual position (i < j)
                    
                    # CASE 1: relation_1 mention appears before relation_2 mention in text
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        
                        # Calculate the index in relation_label list using formula:
                        # For pair (min, max), index = sum of first 'min' terms of series 
                        # + (max - min - 1)
                        # Series: (n-1) + (n-2) + ... where n = num_events
                        # Formula: ((num_events - 1 + num_events - min) * min / 2) - 1 + (max - min)
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        # Validate we found the correct pair
                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal BEFORE relation")

                        # Set label: event_1 BEFORE event_2
                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 1

                    # CASE 2: relation_2 mention appears before relation_1 mention in text
                    # The relation is reversed in our ordered pairs
                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        
                        # Calculate index using same formula
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        # Validate we found the correct pair
                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal BEFORE relation")

                        # Set label: event_1 AFTER event_2 (reverse of BEFORE)
                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 2



    # ========================================================================
    # STEP 7: Process TEMPORAL relations - OVERLAP type
    # ========================================================================
    # OVERLAP relation: events happen at the same time or have overlapping timeframes
    # This is treated the same as SIMULTANEOUS, CONTAINS, ENDS-ON, and BEGINS-ON
    # All map to label value 3 (overlap)
    
    for relation_i in range(len(line_json['temporal_relations']['OVERLAP'])):
        relation_1 = line_json['temporal_relations']['OVERLAP'][relation_i][0]
        relation_2 = line_json['temporal_relations']['OVERLAP'][relation_i][1]

        # Only process event-event relations
        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            # Find all mentions of both events (same logic as BEFORE)
            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            # Apply OVERLAP relation to all mention pairs
            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal OVERLAP relation")

                        # Set label: events overlap (symmetric, so same label both ways)
                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal OVERLAP relation")

                        # Overlap is symmetric, so same label
                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3



    # ========================================================================
    # STEP 8: Process TEMPORAL relations - CONTAINS type
    # ========================================================================
    # CONTAINS: one event's timeframe contains another event's timeframe
    # Treated as OVERLAP (label = 3) for simplification
    
    for relation_i in range(len(line_json['temporal_relations']['CONTAINS'])):
        relation_1 = line_json['temporal_relations']['CONTAINS'][relation_i][0]
        relation_2 = line_json['temporal_relations']['CONTAINS'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal CONTAINS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal CONTAINS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3



    # ========================================================================
    # STEP 9: Process TEMPORAL relations - SIMULTANEOUS type
    # ========================================================================
    # SIMULTANEOUS: events happen at exactly the same time
    # Treated as OVERLAP (label = 3)
    
    for relation_i in range(len(line_json['temporal_relations']['SIMULTANEOUS'])):
        relation_1 = line_json['temporal_relations']['SIMULTANEOUS'][relation_i][0]
        relation_2 = line_json['temporal_relations']['SIMULTANEOUS'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal SIMULTANEOUS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal SIMULTANEOUS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3



    # ========================================================================
    # STEP 10: Process TEMPORAL relations - ENDS-ON type
    # ========================================================================
    # ENDS-ON: one event ends when another event occurs
    # Treated as OVERLAP (label = 3)
    
    for relation_i in range(len(line_json['temporal_relations']['ENDS-ON'])):
        relation_1 = line_json['temporal_relations']['ENDS-ON'][relation_i][0]
        relation_2 = line_json['temporal_relations']['ENDS-ON'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal ENDS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal ENDS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3



    # ========================================================================
    # STEP 11: Process TEMPORAL relations - BEGINS-ON type
    # ========================================================================
    # BEGINS-ON: one event begins when another event occurs
    # Treated as OVERLAP (label = 3)
    
    for relation_i in range(len(line_json['temporal_relations']['BEGINS-ON'])):
        relation_1 = line_json['temporal_relations']['BEGINS-ON'][relation_i][0]
        relation_2 = line_json['temporal_relations']['BEGINS-ON'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal BEGINS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal BEGINS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3



    # ========================================================================
    # STEP 12: Process CAUSAL relations - CAUSE type
    # ========================================================================
    # Causal relations describe cause-effect relationships between events.
    # CAUSE relation: event A causes event B to happen
    #
    # Label encoding:
    # 0 = no causal relation
    # 1 = event_1 causes event_2
    # 2 = event_1 is caused by event_2
    
    for relation_i in range(len(line_json['causal_relations']['CAUSE'])):
        relation_1 = line_json['causal_relations']['CAUSE'][relation_i][0]
        relation_2 = line_json['causal_relations']['CAUSE'][relation_i][1]

        # Only process event-event relations (not time expressions)
        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            # Find all mentions of both events
            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            # Apply CAUSE relation to all mention pairs
            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    
                    # CASE 1: relation_1 appears before relation_2 in text
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in causal CAUSE relation")

                        # Set label: event_1 CAUSES event_2
                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 1

                    # CASE 2: relation_2 appears before relation_1 in text
                    # Need to reverse the causal direction
                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in causal CAUSE relation")

                        # Set label: event_1 is CAUSED BY event_2
                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 2



    # ========================================================================
    # STEP 13: Process CAUSAL relations - PRECONDITION type
    # ========================================================================
    # PRECONDITION: event A must happen before event B can occur
    # This is a type of causal relationship - treated same as CAUSE
    # (event A is a precondition for event B = event A causes event B)
    
    for relation_i in range(len(line_json['causal_relations']['PRECONDITION'])):
        relation_1 = line_json['causal_relations']['PRECONDITION'][relation_i][0]
        relation_2 = line_json['causal_relations']['PRECONDITION'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in causal PRECONDITION relation")

                        # Label as CAUSES (precondition is a form of causation)
                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 1

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in causal PRECONDITION relation")

                        # Label as CAUSED BY
                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 2




    # ========================================================================
    # STEP 14: Process SUBEVENT relations
    # ========================================================================
    # Subevent relations describe hierarchical relationships between events.
    # One event is a component or part of another larger event.
    # Example: "scoring a goal" is a subevent of "playing a match"
    #
    # Label encoding:
    # 0 = no subevent relation
    # 1 = event_1 contains event_2 (event_2 is a subevent of event_1)
    # 2 = event_1 is contained by event_2 (event_1 is a subevent of event_2)
    
    for relation_i in range(len(line_json['subevent_relations'])):
        # Get the parent and child event IDs
        # relation_1 is the parent event that contains relation_2
        relation_1 = line_json['subevent_relations'][relation_i][0]
        relation_2 = line_json['subevent_relations'][relation_i][1]

        # Only process event-event relations
        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT':

            # Find all mentions of both events
            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            # Apply subevent relation to all mention pairs
            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    
                    # CASE 1: parent event mention appears before child event in text
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in subevent relation")

                        # Set label: event_1 CONTAINS event_2
                        article_json['relation_label'][index_in_relation_label]['label_subevent'] = 1

                    # CASE 2: child event mention appears before parent event in text
                    # Need to reverse the containment direction
                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or \
                           article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in subevent relation")

                        # Set label: event_1 is CONTAINED BY event_2
                        article_json['relation_label'][index_in_relation_label]['label_subevent'] = 2


    # ========================================================================
    # STEP 15: Save the processed data to a JSON file
    # ========================================================================
    # Write the complete article_json dictionary to the specified output path
    # This creates a well-structured JSON file with all event and relation labels
    with open(out_path, "w") as f:
        json.dump(article_json, f)

# End of extract_label function




# ============================================================================
# MAIN EXECUTION: Process MAVEN-ERE dataset files
# ============================================================================
# This section applies the extract_label function to all articles in the
# MAVEN-ERE training and validation datasets, converting them from the
# original JSONL format to individual processed JSON files.
# ============================================================================

# ============================================================================
# Process VALIDATION dataset (dev set)
# ============================================================================
# The validation set contains 710 articles for model evaluation
# Input: ./MAVEN_ERE/valid.jsonl (JSONL format - one article per line)
# Output: ./MAVEN_ERE/dev/*.json (individual JSON files per article)

line_id = 0  # Counter to track which line we're processing
with open("./MAVEN_ERE/valid.jsonl", "r") as f:
    # Process exactly 710 articles from the validation set
    while line_id < 710:
        # Read one line (one article) from the JSONL file
        line = f.readline()
        
        # Parse the JSON string into a dictionary
        line_json = json.loads(line)
        
        # Create output path using the article's unique ID
        out_path = "./MAVEN_ERE/dev/" + line_json['id'] + ".json"
        
        # Process this article and save to output file
        extract_label(line_json, out_path)
        
        # Move to next article
        line_id += 1


# ============================================================================
# Process TRAINING dataset
# ============================================================================
# The training set contains 2913 articles for model training
# Input: ./MAVEN_ERE/train.jsonl (JSONL format - one article per line)
# Output: ./MAVEN_ERE/train/*.json (individual JSON files per article)

line_id = 0  # Reset counter for training set
with open("./MAVEN_ERE/train.jsonl", "r") as f:
    # Process exactly 2913 articles from the training set
    while line_id < 2913:
        # Print progress to monitor the processing
        # (Useful since processing 2913 files takes some time)
        print(line_id)
        
        # Read one line (one article) from the JSONL file
        line = f.readline()
        
        # Parse the JSON string into a dictionary
        line_json = json.loads(line)
        
        # Create output path using the article's unique ID
        out_path = "./MAVEN_ERE/train/" + line_json['id'] + ".json"
        
        # Process this article and save to output file
        extract_label(line_json, out_path)
        
        # Move to next article
        line_id += 1


# ============================================================================
# End of preprocessing script
# ============================================================================
# At this point, all articles from both training and validation sets have
# been processed and saved as individual JSON files with structured labels.
# These files are ready to be used for training event extraction models.
# ============================================================================




# Marker to indicate end of script execution
# stop here
