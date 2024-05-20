#%% Import Packages
import re

def search(pattern, string):
    result = re.search(pattern=pattern, string=string)
    if result:
        print(result.group())
        print(result.start())
        print(result.end())
    else:
        print("No match found!")
    return result
        
#%% Beginning of the string match
pattern = "I love"
string = "I love to eat ice-cream!"
result = re.match(pattern=pattern, string=string)
if result:
    print(result.group())
    print(result.start())
    print(result.end())
else:
    print("No match found!")

#%% Anywhere of the string match
pattern = "eat"
string = "I love to eat ice-cream!"
search(pattern=pattern, string=string)

#%% Match any single character in place of period except a newline
pattern = r"l.v"
string = "I love to eat ice-cream!"
search(pattern=pattern, string=string)

#%% Match 0 or more repetition of preceeding character
pattern = r"ea*t"
string = "I love to et ice-cream!"
search(pattern=pattern, string=string)

#%% Match 1 or more repetition of preceeding character
pattern = r"ea+t"
string = "I love to eaat ice-cream!"
search(pattern=pattern, string=string)

#%% Match 0 or 1 repetition of preceeding character
pattern = r"ea?t"
string = "I love to et ice-cream!"
search(pattern=pattern, string=string)

#%% Match any one character inside the bracket
pattern = r"[xyz]"
string = "I love to eat ice-cream!"
search(pattern=pattern, string=string)

#%% Match exactly the number of repetition of preceeding character
pattern = r"ea{2}t" # number of rep is exactly 2
string = "I love to eaat ice-cream!"
search(pattern=pattern, string=string)
pattern = r"ea{2,}t" # number of rep is at least 2
string = "I love to eaaaaat ice-cream!"
search(pattern=pattern, string=string)
pattern = r"ea{2,4}t" # number of rep is between 2 to 4
string = "I love to eat ice-cream!"
search(pattern=pattern, string=string)

#%% Match the start of the string
pattern = r"^I love"
string = "I love to eat ice-cream!"
search(pattern=pattern, string=string)

#%% Match the end of the string
pattern = r"cream!$"
string = "I love to eat ice-cream!"
search(pattern=pattern, string=string)

#%% Match either the pattern before or after the bar
pattern = r"eat|feed"
string = "I love to feed ice-cream!"
search(pattern=pattern, string=string)

#%% Match the special characters
pattern = r"[0-9]" # digits (shortcut: \d)
string = "1234abcd"
search(pattern=pattern, string=string)
pattern = r"[a-zA-Z]" # letters
string = "1234abcd"
search(pattern=pattern, string=string)
pattern = r"[ \t\n\r\f\v]" # whitespaces (shortcut: \s)
string = "1234 abcd"
search(pattern=pattern, string=string)
pattern = r"[^a-zA-Z0-9]" # special characters
string = "1234@abcd"
search(pattern=pattern, string=string)
pattern = r"[a-zA-Z0-9_]" # alpha numeric with underscore only (shortcut: \w)
string = "_1234_abcd"
search(pattern=pattern, string=string)
pattern = r"[^0-9]" # complement digits (shortcut: \D)
string = "1234abcd"
search(pattern=pattern, string=string)
pattern = r"\beat\b" # match whole words in start and end of word boundary
string = "I eat rice"
search(pattern=pattern, string=string)

#%% Match group using parenthesis
pattern = r"(ab)+"
string = "ababab"
search(pattern=pattern, string=string)

#%% Create groups 
pattern = r"(\w+)@([\w\.]+)"
string = "contact us at support@email.com"
result = search(pattern=pattern, string=string)
print(result.group(0))
print(result.group(1))
print(result.group(2))

#%% Create named groups 
pattern = r"(?P<username>\w+)@(?P<domain>[\w\.]+)"
string = "contact us at support@email.com"
result = search(pattern=pattern, string=string)
print(result.group(0))
print(result.group("username"))
print(result.group("domain"))

#%% Greedy and Non greedy search
pattern = r"<.*>" # Greedy
string = r"<h1>TITLE</h1>"
result = search(pattern=pattern, string=string)
pattern = r"<.*?>" # Non greedy so as soon as > is found, it stops the search
string = r"<h1>TITLE</h1>"
result = search(pattern=pattern, string=string)

#%% Compile most frequently used reg exp pattern
pattern = re.compile("(\w+)")
string = "Hello World!"
pattern.search(string).group()

#%% Change behaviour of search
""" 1. re.IGNORECASE: Makes the pattern case-insensitive.
	2. re.MULTILINE: Changes the behavior of ^ and $ to match the start and end of each line within the string, rather than just the start and end of the string.
	3. re.DOTALL: Makes the dot (.) match any character, including a newline.
    4. re.VERBOSE: Allows you to write more readable regular expressions by ignoring whitespace and permitting comments within the pattern.
 Combine flags using |
 """
 
pattern = r"""
    hello   # match hello
    \s+     # one or more whitespace characters
    world   # match world
"""
string = "hello   world"
re.search(pattern, string, re.VERBOSE).group()

#%% Find multiple occurences
pattern = r"\d"
string = "Let's find out reason 2 and 3"
print(re.findall(pattern, string))
print(list(re.finditer(pattern, string))) # gives an iterator which has more information

#%% Replace string
pattern = r"\d"
string = "Let's find out reason 2 and 3"
print(re.sub(pattern, "blah", string)) # all occ are replaced

#%% Split string
pattern = r"(\s)" # capturing group so it retains the group in split text
string = "Hello, world! How are you?"
print(re.findall(pattern, string))
res = re.split(pattern, string)
print(res)

pattern = r"[\s]" # class group so it removes the group in split text
string = "Hello, world! How are you?"
print(re.findall(pattern, string))
res = re.split(pattern, string)
print(res)

# %%
