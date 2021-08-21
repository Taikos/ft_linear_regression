#!/usr/bin/env python
# -*-coding: utf-8 -*

import sys
import re
import pandas as pd

from predict import km_predict, normalize_km, denormalize_price
from predict import query_yes_no

# Bonus - to visualize the data as graph.
import matplotlib.pyplot as plt

START_THETA_0 = float(0)
START_THETA_1 = float(0)
LEARNING_RATE = float(0.0001)
ITERATIONS = 2000
ERROR_LOSS_HISTORY = []

### BONUS ####
def data_grapher(theta_0, theta_1, data_file):
    # File values and Linear regression representation
    xx = []
    yy = []
    for ind in data_file.index:
        xx.append(data_file["km"][ind])
        yy.append(data_file["price"][ind])
    gradient_x = [float(min(data_file["km"])), float(max(data_file["km"]))]
    gradient_y = []
    for gradient_price in gradient_x:
        gradient_price = (
            theta_1 * normalize_km(data_file["km"], gradient_price) + theta_0
        )
        gradient_y.append(denormalize_price(data_file["price"], gradient_price))
    axes = plt.axes()
    axes.grid()
    plt.title("Real values and Linear regression")
    plt.xlabel("Km")
    plt.ylabel("Price")
    plt.plot(xx, yy, "ro", gradient_x, gradient_y, "b-")
    plt.show()


def cost_grapher(cost_monitor):
    # Cost function representation
    xx = []
    yy = []
    for i in range(len(cost_monitor)):
        xx.append(i)
        yy.append(cost_monitor[i])
    axes = plt.axes()
    axes.grid()
    plt.xlabel("Iteration n°")
    plt.ylabel("Global Error Cost")
    plt.scatter(xx, yy)
    plt.show()


def cost_calculator(theta_0, theta_1, m, kms, prices):
    # Cost function calculation: J(theta_0,theta_1) = 1/2m * (sum{i=0}to{m} (theta_0 + theta_1  x_i - y_i)^2)
    overall_cost = float(0)
    for i in range(0, m):
        tmp_cost = float(
            ((theta_0 + (theta_1 * float(kms[i]))) - float(prices[i])) ** 2
        )
        overall_cost += tmp_cost
    return 1 / (2 * m) * overall_cost


### END BONUS ####


def boldDriver(loss_number, theta_0, theta_1, d_theta_0, d_theta_1, learning_rate, m):
    # Compare error loss of previous ite with the new one
    up_learning_rate = learning_rate
    if len(ERROR_LOSS_HISTORY) > 1:
        if loss_number < ERROR_LOSS_HISTORY[-1]:
            up_learning_rate *= 1.05
        else:
            theta_0 += d_theta_0 / m * learning_rate
            theta_1 += d_theta_1 / m * learning_rate
            up_learning_rate *= 0.5
    return [theta_0, theta_1, up_learning_rate]


def errorLossIdentification(theta_0, theta_1, m, kms, prices):
    # Error = (y actual — y pred)²
    error_loss = float(0)
    for i in range(0, m):
        error_loss += (prices[i] - (theta_1 * kms[i] + theta_0)) ** 2
    error_loss = error_loss / m
    return error_loss


def partial_derivatives(theta_0, theta_1, m, kms, prices):
    # Sum of (theta_0 + theta_1 * kms[i]) - prices[i]
    # Sum of ((theta_0 + theta_1 * kms[i]) - prices[i]) * kms[i]
    tmp_0 = float(0)
    tmp_1 = float(0)
    for i in range(0, m):
        tmp_0 += (theta_0 + theta_1 * float(kms[i])) - float(prices[i])
        tmp_1 += ((theta_0 + theta_1 * float(kms[i])) - float(prices[i])) * float(
            kms[i]
        )
    return [tmp_0, tmp_1]


def theta_update(theta_0, theta_1, m, kms, prices, learning_rate):
    # Apply cost function with gradient descent algorithm
    [derive_theta_0, derive_theta_1] = partial_derivatives(
        theta_0, theta_1, m, kms, prices
    )
    # Update the current values of thetas
    theta_0 -= derive_theta_0 / m * learning_rate
    theta_1 -= derive_theta_1 / m * learning_rate
    # End of the gradient descent algorithm

    # Adapt learning rate during each iteration
    error_loss_number = errorLossIdentification(theta_0, theta_1, m, kms, prices)
    [theta_0, theta_1, learning_rate] = boldDriver(
        error_loss_number,
        theta_0,
        theta_1,
        derive_theta_0,
        derive_theta_1,
        learning_rate,
        m,
    )
    ERROR_LOSS_HISTORY.append(error_loss_number)

    return [theta_0, theta_1, learning_rate]


def normalizeData(elems):
    data = []
    min_elem = min(elems)
    max_elem = max(elems)
    for elem in elems:
        data.append((elem - min_elem) / (max_elem - min_elem))
    return data


def train_model():
    try:
        extract_info = pd.read_csv("data.csv", sep=",")
    except:
        print("data.csv file not found please add it to the project folder")
        sys.exit(-1)
    kms = extract_info.iloc[0 : len(extract_info), 0]
    prices = extract_info.iloc[0 : len(extract_info), 1]
    data_len = len(kms)
    tmp_theta_0 = START_THETA_0
    tmp_theta_1 = START_THETA_1
    print("First theta 0 to start: " + str(tmp_theta_0))
    print("First theta 1 to start: " + str(tmp_theta_1))
    cost_monitor = []
    tmp_learning_rate = LEARNING_RATE
    kms = normalizeData(kms)
    prices = normalizeData(prices)
    for i in range(0, ITERATIONS):
        [tmp_up_theta_0, tmp_up_theta_1, tmp_up_learning_rate] = theta_update(
            tmp_theta_0, tmp_theta_1, data_len, kms, prices, tmp_learning_rate
        )
        tmp_theta_0 = tmp_up_theta_0
        tmp_theta_1 = tmp_up_theta_1
        tmp_learning_rate = tmp_up_learning_rate
        print("New theta 0 set: " + str(tmp_theta_0))
        print("New theta 1 set: " + str(tmp_theta_1))
        print("Updated learning rate: " + str(tmp_learning_rate))
        # Bonus - Store data for the learning phase graph - apply cost function
        cost_monitor.append(
            cost_calculator(tmp_theta_0, tmp_theta_1, data_len, kms, prices)
        )

    # Bonus - Visualize learning phase - Minimise cost function
    cost_grapher(cost_monitor)

    # Bonus - Visualize linear regression and data
    data_grapher(tmp_theta_0, tmp_theta_1, extract_info)

    return [tmp_theta_0, tmp_theta_1]


def main():
    if len(sys.argv) != 1:
        print("ERROR 1 - Wrong filename.\n- usage: python3 train_and_predict.py")
        return -1
    if sys.argv[0]:
        print("Start training model\n")
        [final_theta_O, final_theta_1] = train_model()
        print(
            "After "
            + str(ITERATIONS)
            + ", A new formula is ready to be used with theta O = "
            + str(final_theta_O)
            + " and theta 1 = "
            + str(final_theta_1)
        )
        want_a_prediction = query_yes_no("Do you want a new price estimation?", "yes")
        if want_a_prediction:
            sys.stdout.write("What is the mileage on the car (km)?\n")
            km = input()
            num_format = re.compile(r"^[0-9]*[.]{0,1}[0-9]*$")

            while re.match(num_format, km) is None:
                sys.stdout.write(
                    "Please enter a positive mileage using int type or float type\n"
                )
                km = input()
            km_predict(km, final_theta_O, final_theta_1)

            return -1
        else:
            print("You don't want an estimation! Come back when you want.")
            return -1
    else:
        print("ERROR 1 - Wrong filename.\n- usage: python3 train_and_predict.py")
        return -1


if __name__ == "__main__":
    main()
