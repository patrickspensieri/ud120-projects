#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    print "type(locals()) : ", type(locals())
    # iterate through al function arguments
    # for arg in locals():
    #     print "type(arg) : ", type(arg)
    errors = (net_worths - predictions)**2
    cleaned_data = zip(ages, net_worths, errors)
    # lambda function set key to column 2, sort by error
    # param reverse=True sets to descending
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2], reverse=False)
    # everything except the last 9 items in list
    cleaned_data = cleaned_data[:-9]

    return cleaned_data

