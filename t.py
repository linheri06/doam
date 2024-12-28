from predict_t import PredictionTool
def main():
    data = "input_series.csv"
    model = "final_model.pth"
    #PredictionTool(data, model).predict()
    print(PredictionTool(data, model).predict())

if __name__ == "__main__":
    main()
