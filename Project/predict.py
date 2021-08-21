#!/usr/bin/env python
# -*-coding: utf-8 -*

import sys
import re
import pandas as pd

THETA_0_0 = float(0)
THETA_0_1 = float(0)


def normalize_km(kms, km):
    # The data is scaled to a fixed range to create a common scale between KMs and Prices.
    return (km - min(kms)) / (max(kms) - min(kms))


def denormalize_price(prices, price):
    # The data is unscaled to the real range to provide the right price.
    return (price * (max(prices) - min(prices))) + min(prices)


def km_predict(km, theta_0, theta_1):
    # Formula applied: prixEstime(kilométrage) = θ0 + (θ1 ∗ kilométrage).
    try:
        extract_info = pd.read_csv("data.csv", sep=",")
    except:
        print("data.csv file not found please add it to the project folder")
        sys.exit(-1)
    if float(theta_0) and float(theta_1) != 0:
        price_prediction = (
            theta_1 * normalize_km(extract_info["km"], float(km)) + theta_0
        )
        print(
            "Based on the mileage, the price estimation is\n"
            + str("{:.2f}".format(denormalize_price(extract_info["price"], price_prediction)))
            + "$"
        )
    else:
        print(
            "Based on the mileage, the price estimation is\n"
            + str(float(theta_0) + (float(theta_1) * float(km)))
            + "$"
        )


def query_yes_no(question, default):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def main():
    if len(sys.argv) != 1:
        print("ERROR 1 - Wrong filename.\n- usage: python3 predict.py")
        return -1
    if sys.argv[0]:
        want_a_prediction = query_yes_no("Do you want a new price estimation?", "yes")
        if want_a_prediction == True:
            sys.stdout.write("What is the mileage on the car (km)?\n")
            km = input()
            num_format = re.compile(r"^[0-9]*[.]{0,1}[0-9]*$")

            while re.match(num_format, km) == None:
                sys.stdout.write(
                    "Please enter a positive mileage using int type or float type\n"
                )
                km = input()
            km_predict(km, THETA_0_0, THETA_0_1)
            return -1
        else:
            print("You don't want an estimation! Come back when you want.")
            return -1
    else:
        print("ERROR 1 - Wrong filename.\n- usage: python3 predict.py")
        return -1


if __name__ == "__main__":
    main()
