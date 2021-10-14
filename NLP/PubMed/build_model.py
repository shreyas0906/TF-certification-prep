import os

data_dir = "pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/"


def get_lines(filename):
    with open(filename,'r') as f:
        return f.readlines()


train_lines = get_lines(data_dir + 'train.txt')
print(len(train_lines))


def preprocess_text_with_line_numbers(filename):
    """Returns a list of dictionaries of abstract line data.

    Takes in filename, reads its contents and sorts through each line,
    extracting things like the target label, the text of the sentence,
    how many sentences are in the current abstract and what sentence number
    the target line is.

    Args:
        filename: a string of the target text file to read and extract line data
        from.

    Returns:
        A list of dictionaries each containing a line from an abstract,
        the lines label, the lines position in the abstract and the total number
        of lines in the abstract where the line is from. For example:

        [{"target": 'CONCLUSION',
          "text": The study couldn't have gone better, turns out people are kinder than you think",
          "line_number": 8,
          "total_lines": 8}]
    """
    input_lines = get_lines(filename)  # get all lines from filename
    abstract_lines = ""  # create an empty abstract
    abstract_samples = []  # create an empty list of abstracts

    # Loop through each line in target file
    for line in input_lines:
        if line.startswith("###"):  # check to see if line is an ID line
            abstract_id = line
            abstract_lines = ""  # reset abstract string
        elif line.isspace():  # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines()  # split abstract into separate lines

            # Iterate through each line in abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}  # create empty dict to store data from line
                target_text_split = abstract_line.split("\t")  # split target label from text
                line_data["target"] = target_text_split[0]  # get target label
                line_data["text"] = target_text_split[1].lower()  # get target text and lower it
                line_data["line_number"] = abstract_line_number  # what number line does the line appear in the abstract?
                line_data["total_lines"] = len(abstract_line_split) - 1  # how many total lines are in the abstract? (start from 0)
                abstract_samples.append(line_data)  # add line data to abstract samples list

        else:  # if the above conditions aren't fulfilled, the line contains a labelled sentence
            abstract_lines += line

    return abstract_samples


train_samples = preprocess_text_with_line_numbers(data_dir + 'train.txt')
test_samples = preprocess_text_with_line_numbers(data_dir + 'test.txt')
val_samples = preprocess_text_with_line_numbers(data_dir + 'dev.txt')

print(f"train samples: {len(train_samples)}\t val_samples: {len(val_samples)}\t test_samples: {len(test_samples)}")

print(train_samples[:10])