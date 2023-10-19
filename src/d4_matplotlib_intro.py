# Importing the libraries
import numpy as np
from matplotlib import pyplot as plt


# Importing the dataset
def create_data():
    file_name = "day6_matplotlib_intro.csv"
    print("file_name: ", file_name)
    raw_data = open(file_name, "rt")
    # load txt defaults to floats
    data = np.loadtxt(raw_data, delimiter=",", dtype="str")
    header = data[0:1, :]
    data = data[1:, :].astype("float")

    return header, data


# noinspection PyUnusedLocal
def ex1_company_profit(header, data):
    plt.plot(data[:, 0], data[:, 8])
    plt.xlabel("Month number")
    plt.ylabel("Profit in dollar")
    plt.title("Company profit per month")
    plt.xticks(data[:, 0])
    plt.show()


# noinspection PyUnusedLocal
def ex2_company_profit_style(header, data):
    plt.plot(
        data[:, 0],
        data[:, 8],
        linestyle="dashed",
        linewidth="3",
        color="red",
        label="Profit data of last year",
        marker="o",
        markersize=10,
        markerfacecolor="black",
        markeredgewidth=2,
    )
    plt.legend(loc="lower right")
    plt.xlabel("Month number")
    plt.ylabel("Profit in dollar")
    plt.title("Company sales data of last year")
    plt.xticks(data[:, 0])
    plt.show()


# noinspection PyUnusedLocal
def ex3_company_profit_style(header, data):
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
    ]
    names = [
        "Face Cream",
        "Face Wash",
        "Toothpaste",
        "Bashing Soap",
        "Shampoo",
        "Moisturizer",
    ]

    for idx, name in enumerate(names):
        plt.plot(
            data[:, 0],
            data[:, idx + 1],
            linestyle="-",
            linewidth="3",
            color=colors[idx],
            label=f"{name} Sales Data",
            markersize=10,
            marker="o",
        )

    plt.xticks(data[:, 0])
    plt.legend()
    plt.ylabel("Sales units in number")
    plt.xlabel("Month Number")
    plt.show()


# noinspection PyUnusedLocal
def ex4_tooth_paste_grid(header, data):
    plt.scatter(data[:, 0], data[:, 3], label="Toothpaste Sales data")
    plt.legend()
    plt.xlabel("Number of units sold")
    plt.ylabel("Month Number")
    plt.title("Toothpaste Sales data")
    plt.grid(linestyle="--")
    plt.xticks(data[:, 0])
    plt.show()


def ex5_bar_chart(header, data):
    width = 0.35
    plt.bar(
        data[:, 0],
        data[:, 1],
        width=width,
        label="Face cream sales data",
    )
    plt.bar(
        data[:, 0] + width,
        data[:, 2],
        width=width,
        label="Face wash sales data",
    )
    plt.xticks(data[:, 0])
    plt.grid(ls="--")
    plt.xlabel("Month Number")
    plt.ylabel("Sales in number")
    plt.title("Face wash and face cream sales data")
    plt.legend(loc="upper left")
    plt.show()


def _main():
    header, data = create_data()
    ex1_company_profit(header, data)
    ex2_company_profit_style(header, data)
    ex3_company_profit_style(header, data)
    ex4_tooth_paste_grid(header, data)
    ex5_bar_chart(header, data)


if __name__ == "__main__":
    _main()
