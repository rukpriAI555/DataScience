import pandas as pd
import numpy as np


class Univariate:

    # 1. Separate Quantitative and Qualitative Columns
    def QuanQual(dataset):

        quan = []
        qual = []

        for columnName in dataset.columns:
            if dataset[columnName].dtypes == 'O':
                qual.append(columnName)
            else:
                quan.append(columnName)

        return quan, qual

    # 2. Frequency Table
    def FreqTable(columnName, dataset):

        FreqTable = pd.DataFrame(
            columns=[
                "Unique_values",
                "Frequency",
                "Relative Frequency",
                "Cumsum"
            ]
        )

        FreqTable["Unique_values"] = dataset[columnName].value_counts().index

        FreqTable["Frequency"] = dataset[columnName].value_counts()

        FreqTable["Relative Frequency"] = (
            FreqTable["Frequency"] / len(dataset)
        )

        FreqTable["Cumsum"] = (
            FreqTable["Relative Frequency"].cumsum()
        )

        return FreqTable

    # 3. Univariate Descriptive Analysis
    def univariate(dataset, quan):

        descriptive = pd.DataFrame(
            index=[
                "Mean",
                "Median",
                "Mode",
                "Q1:25%",
                "Q2:50%",
                "Q3:75%",
                "99%",
                "Q4:100%",
                "IQR",
                "1.5Rule",
                "Lesser",
                "Greater",
                "Min",
                "Max",
                "kurtosis",
                "skew",
                "var",
                "stddev"
            ],
            columns=quan
        )

        for columnName in quan:

            # Mean
            descriptive.loc["Mean", columnName] = (
                dataset[columnName].mean()
            )

            # Median
            descriptive.loc["Median", columnName] = (
                dataset[columnName].median()
            )

            # Mode
            descriptive.loc["Mode", columnName] = (
                dataset[columnName].mode()[0]
            )

            # First Quartile
            descriptive.loc["Q1:25%", columnName] = (
                dataset[columnName].describe()["25%"]
            )

            # Second Quartile / Median
            descriptive.loc["Q2:50%", columnName] = (
                dataset[columnName].describe()["50%"]
            )

            # Third Quartile
            descriptive.loc["Q3:75%", columnName] = (
                dataset[columnName].describe()["75%"]
            )

            # 99th Percentile
            descriptive.loc["99%", columnName] = (
                np.percentile(
                    dataset[columnName],
                    99
                )
            )

            # Maximum / 100th Percentile
            descriptive.loc["Q4:100%", columnName] = (
                dataset[columnName].describe()["max"]
            )

            # IQR
            descriptive.loc["IQR", columnName] = (
                descriptive.loc["Q3:75%", columnName]
                - descriptive.loc["Q1:25%", columnName]
            )

            # 1.5 Rule
            descriptive.loc["1.5Rule", columnName] = (
                1.5 * descriptive.loc["IQR", columnName]
            )

            # Lower Boundary
            descriptive.loc["Lesser", columnName] = (
                descriptive.loc["Q1:25%", columnName]
                - descriptive.loc["1.5Rule", columnName]
            )

            # Upper Boundary
            descriptive.loc["Greater", columnName] = (
                descriptive.loc["Q3:75%", columnName]
                + descriptive.loc["1.5Rule", columnName]
            )

            # Minimum
            descriptive.loc["Min", columnName] = (
                dataset[columnName].min()
            )

            # Maximum
            descriptive.loc["Max", columnName] = (
                dataset[columnName].max()
            )

            # Kurtosis
            descriptive.loc["kurtosis", columnName] = (
                dataset[columnName].kurtosis()
            )

            # Skewness
            descriptive.loc["skew", columnName] = (
                dataset[columnName].skew()
            )

            # Variance
            descriptive.loc["var", columnName] = (
                dataset[columnName].var()
            )

            # Standard Deviation
            descriptive.loc["stddev", columnName] = (
                dataset[columnName].std()
            )

        return descriptive

    # 4. Finding Outliers
    def Finding_outliers(descriptive, quan):

        lesser = []
        greater = []

        for columnName in quan:

            # Check lower outlier
            if (
                descriptive[columnName]["Min"]
                < descriptive[columnName]["Lesser"]
            ):
                lesser.append(columnName)

            # Check upper outlier
            if (
                descriptive[columnName]["Max"]
                > descriptive[columnName]["Greater"]
            ):
                greater.append(columnName)

        return lesser, greater

    # 5. Handle / Replace Outliers
    def Handle_outliers(dataset, descriptive, quan):

        Lesser, Greater = Univariate.Finding_outliers(
            descriptive,
            quan
        )

        # Replace lower outliers
        for ColumnName in Lesser:

            dataset.loc[
                dataset[ColumnName]
                < descriptive[ColumnName]["Lesser"],
                ColumnName
            ] = descriptive[ColumnName]["Lesser"]

        # Replace upper outliers
        for ColumnName in Greater:

            dataset.loc[
                dataset[ColumnName]
                > descriptive[ColumnName]["Greater"],
                ColumnName
            ] = descriptive[ColumnName]["Greater"]

        return dataset
