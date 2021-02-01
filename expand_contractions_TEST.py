import re
from contractions import CONTRACTION_MAP


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

print(expand_contractions("I shouldn't've put the keys under the carpet."))

# assert expand_contractions("That ain't true.") == 'That is not true.', 'That is not true.'
assert expand_contractions("Your actions now, doesn't change anything you've done in your past.") == "Your actions now, does not change anything you have done in your past."
assert expand_contractions("I shouldn't have put the keys under the carpet.") == "I should not have put the keys under the carpet."
assert expand_contractions("It wasn't meant for you.") == "It was not meant for you."
assert expand_contractions("I'll be available.") == "I will be available."

print("[INFO]: TESTS PASSED SUCCESSFULLY")