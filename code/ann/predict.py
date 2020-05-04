model = Regressor2()


def predict()
    model.load_state_dict(torch.load('best_model/modelHeston.pt',map_location=torch.device('cpu')))