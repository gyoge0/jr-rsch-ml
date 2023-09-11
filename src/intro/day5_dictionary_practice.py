# =============================================================================
# Create a dictionary containing "name", "city", and "cake" for
# "John" from "Westminster" who likes "Chocolate".
# Display the dictionary.
# =============================================================================
person = {
    "name": "John",
    "city": "Westminster",
    "cake": "Chocolate",
}
print(person)

# =============================================================================
# Delete the entry for "cake".
# Display the dictionary.
# =============================================================================
del person["cake"]

# =============================================================================
# Add an entry for "fruit" with "Mango" and display the dictionary.
# =============================================================================
person["fruit"] = "Mango"

# =============================================================================
# Display the dictionary keys.
# =============================================================================
print(*person.keys(), sep=", ")

# =============================================================================
# Display the dictionary values.
# =============================================================================
print(*person.values(), sep=", ")

# =============================================================================
# Display whether or not "cake" is a key in the dictionary (i.e. False) (now).
# Display whether or not "Mango" is a value in the dictionary (i.e. True).
# =============================================================================
print("cake" in person.keys())
print("Mango" in person.values())

# =============================================================================
# Using the dictionary from item 1: Make a dictionary using the same keys
# but with the number of ‘t’s in each value.
# =============================================================================
person_ts = {k: v.count("t") for k, v in person.items()}
