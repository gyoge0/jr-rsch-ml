# Print Grades Function
def print_grades(in_grades_dict):
    for key in sorted(in_grades_dict.keys()):
        print(key, in_grades_dict[key])


# Count Grades Function goes below
def count_grades(grades_list):
    return {g: grades_list.count(g) for g in set(grades_list)}


# End Functions
def _main():
    #  Functions called by the code below
    # Make sure you initialize them as floats!
    grades_list = ["B", "C", "A", "A", "D", "B", "B", "A", "C"]
    counts_dict = count_grades(grades_list)
    print_grades(counts_dict)


if __name__ == "__main__":
    _main()
