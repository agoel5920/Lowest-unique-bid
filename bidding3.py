import numpy as np
from scipy.optimize import curve_fit

# Data
bids = [718200, 547800, 404600, 399000, 389400, 720500, 357100, 289300, 562300, 308000,
        410000, 492900, 436600, 304400, 386000, 311600, 215900, 370300, 310200, 423500,
        256600, 289600, 249400, 296800, 518400, 309300, 215200, 379100, 352400, 247500,
        406100]
          
unique_bids = [7068, 3640, 5365, 1556, 4818, 7303, 2563, 3386, 3667, 1658, 2860,
               4708, 1813, 3081, 1796, 3463, 3037, 4609, 2867, 4515, 1353, 2925,
               3660, 1292, 3602, 3757, 3028, 4076, 4562, 1393]

def predict_unique_bid(bids, unique_bids, poly_degree):

  # Calculate moving averages  
  moving_avgs = []
  for i in range(len(bids)-2):
    moving_avgs.append((bids[i] + bids[i+1] + bids[i+2]) / 3)

  # Fit polynomial curve
  xdata = np.array(list(range(1, len(moving_avgs)+1)))
  ydata = np.array(moving_avgs)

  def poly_func(x, *params):
    y = 0
    for i, param in enumerate(params):
      y += param * x**i

    return y

  params, covar = curve_fit(poly_func, xdata, ydata, p0=[1]*poly_degree)

  # Make prediction
  x_new = len(moving_avgs) + 1
  y_pred = poly_func(x_new, *params)

  # Get ratio and scale unique bid
  ratio = bids[-1] / y_pred
  last_unique = unique_bids[-3]
  prediction = last_unique * ratio
  
  return int(prediction)

  # Example
pred = predict_unique_bid(bids, unique_bids, poly_degree=3)
print("Predicted unique bid for degree=3:", pred)

pred = predict_unique_bid(bids, unique_bids, poly_degree=2)
print("Predicted unique bid for degree=2:", pred)

pred = predict_unique_bid(bids, unique_bids, poly_degree=1)
print("Predicted unique bid for degree=1:", pred)