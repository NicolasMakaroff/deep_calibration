model = Regressor2()


def predict_vol_map(model,normalize=False):
    S = tqdm(np.arange(0.5,1.6,0.1), desc="S")
    for i in S:
        T = tqdm([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0], desc="tau")
        for j in [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0]:
            inp = torch.tensor([i,j,0.05,-0.05,1.5,0.1,0.3,0.1])
            reg = model(inp)
            input_.append([reg]) 
    input_ = np.array(input_)
    input_ = np.reshape(input_,(11,8))
    return input_