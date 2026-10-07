from math import exp

if __name__ == '__main__':
    epochs = 20
    full_weight_epoch = epochs - epochs // 3
    steep = 3.0

    for epoch in range(1, epochs+1):

        prog = min(
            (epoch - 1) / (full_weight_epoch - 1),
            1.0
        )

        # beta = (1 - exp(-steep * prog)) / (1 - exp(-steep))
        beta = (exp(steep * prog) - 1) / (exp(steep) - 1)
        
        print(f'Weight used in epoch: {epoch} - {beta:.4f}')