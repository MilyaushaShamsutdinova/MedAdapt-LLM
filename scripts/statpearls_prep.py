import os
import json
import tqdm
import xml.etree.ElementTree as ET
from datasets import Dataset


def ends_with_ending_punctuation(s):
    """
    Return True if the string ends with one of '.', '?' or '!'
    """
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    """
    Concatenate a title and content into one string.
    Ensures there is exactly one sentence-ending punctuation before the content.
    """
    title = title.strip()
    content = content.strip()
    if ends_with_ending_punctuation(title):
        return title + " " + content
    else:
        return title + ". " + content

def extract_text(element):
    """
    Recursively extract all text from an XML element, including its children and tails.
    Returns the combined, stripped text.
    """
    text = (element.text or "").strip()

    for child in element:
        text += (" " if len(text) else "") + extract_text(child)
        if child.tail and child.tail.strip():
            text += (" " if len(text) else "") + child.tail.strip()
    return text.strip()

def is_subtitle(element):
    """
    Determine if an XML element is acting as a subtitle:
    - It must be a <p> tag
    - It must contain exactly one child, a <bold> tag
    - The <bold> tag must have no tail text
    """
    if element.tag != 'p':
        return False
    children = list(element)
    if len(children) != 1 or children[0].tag != 'bold':
        return False
    # If the bold tag has trailing text, it's not a pure subtitle
    if children[0].tail and children[0].tail.strip():
        return False
    return True

def extract(fpath):
    """
    Parse a .nxml file, walk through its sections, paragraphs, and lists,
    and build a list of JSON strings where each record has:
      - id: unique identifier
      - title: hierarchical title (article title -- section title [-- subsection title])
      - content: the raw text chunk
      - contents: title + content, concatenated into one string
    Small consecutive paragraphs (<200 chars) or list items are merged
    if the combined length remains <1000 chars.
    """
    fname = os.path.basename(fpath).replace(".nxml", "")
    tree = ET.parse(fpath)
    title = tree.getroot().find(".//title").text
    sections = tree.getroot().findall(".//sec")

    saved_text = []
    j = 0
    last_text = None
    last_json = None
    last_node = None

    for sec in sections:
        sec_title = sec.find('./title').text.strip()
        sub_title = ""
        prefix = " -- ".join([title, sec_title])
        last_text = None
        last_json = None
        last_node = None

        for ch in sec:
            if is_subtitle(ch):
                last_text = None
                last_json = None
                sub_title = extract_text(ch)
                prefix = " -- ".join(prefix.split(" -- ")[:2] + [sub_title])

            elif ch.tag == 'p':
                curr_text = extract_text(ch)
                if len(curr_text) < 200 and last_text is not None and len(last_text + curr_text) < 1000:
                    last_text = " ".join([last_json['content'], curr_text])
                    last_json['content'] = last_text
                    last_json['contents'] = concat(last_json['title'], last_text)
                    saved_text[-1] = json.dumps(last_json)
                else:
                    last_text = curr_text
                    last_json = {
                        "id": '_'.join([fname, str(j)]),
                        "title": prefix,
                        "content": curr_text
                    }
                    last_json["contents"] = concat(prefix, curr_text)
                    saved_text.append(json.dumps(last_json))
                    j += 1

            elif ch.tag == 'list':
                list_text = [extract_text(c) for c in ch]
                combined_list = " ".join(list_text)

                if last_text is not None and len(combined_list + last_text) < 1000:
                    last_text = " ".join([last_json["content"]] + list_text)
                    last_json['content'] = last_text
                    last_json['contents'] = concat(last_json['title'], last_text)
                    saved_text[-1] = json.dumps(last_json)

                elif len(combined_list) < 1000:
                    last_text = combined_list
                    last_json = {
                        "id": '_'.join([fname, str(j)]),
                        "title": prefix,
                        "content": combined_list
                    }
                    last_json["contents"] = concat(prefix, combined_list)
                    saved_text.append(json.dumps(last_json))
                    j += 1

                else:
                    last_text = None
                    last_json = None
                    for c in list_text:
                        record = {
                            "id": '_'.join([fname, str(j)]),
                            "title": prefix,
                            "content": c,
                            "contents": concat(prefix, c)
                        }
                        saved_text.append(json.dumps(record))
                        j += 1

                if last_node is not None and is_subtitle(last_node):
                    sub_title = ""
                    prefix = " -- ".join([title, sec_title])

            last_node = ch
    return saved_text


if __name__ == "__main__":
    # Retrieve all .nxml files
    dataset_dir = "data/statpearls/statpearls_NBK430685"
    all_data = []
    fnames = sorted([fname for fname in os.listdir(dataset_dir) if fname.endswith(".nxml")])

    # Loop through each file, extract its text chunks, and accumulate
    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join(dataset_dir, fname)
        saved_text = extract(fpath)
        for json_str in saved_text:
            record = json.loads(json_str)
            all_data.append(record)

    # Create a Hugging Face Dataset from the list of records
    hf_dataset = Dataset.from_list(all_data)

    # Save the dataset
    hf_dataset.save_to_disk("data/statpearls/MedRAG_statpearls")
    hf_dataset.push_to_hub("MilyaShams/MedRAG_statpearls", private=False)

