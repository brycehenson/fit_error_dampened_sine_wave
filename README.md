# Fit Error for Damped Sine Wave
by **[Bryce M. Henson](https://github.com/brycehenson)**


This repository derives the error in the parameters when fitting a damped sine wave.
```math
    f(t)=A e^{-t\lambda} \sin{(\omega t+\phi)}
```
Given a fit to N observations evenly spaced over obeservation time T where each sample has noise $\sigma_m$ we find a closed form expression the uncertainty in the parameters $A$, $\lambda$ , $\omega$ , $\phi$.

## Motivation

Dampened sine waves are ubiquitous across physics and engineering appearing in everything from mechanical oscillators  to the Rabi oscillations of a two level system.
Expressions for the uncertainty in these fit partameters would be useful as both a predictive tool, which can be used to estimate the performance of a measurement, and as a diagnostic to understand if a measurement is limited by anticipated noise sources.

## Results
We derive the following expresions for the uncertainty.

Amplitude
```math
\sigma_A^2 = \frac{\sigma_m^{2}}{N}
    \frac{
        4 T \lambda \left(2 T^{2} \lambda^{2} + 2 T \lambda - e^{2 T \lambda} + 1\right)
    }{
        \left(2 T^{2} \lambda^{2} - \cosh{\left(2 T \lambda \right)} + 1\right)
    }
```

Phase
```math
    \sigma_\phi^2 = \frac{\sigma_m^{2}}{N  A^{2}}
    \frac{
        4 T \lambda
        \left(
            2 T^{2} \lambda^{2} +2  T \lambda - e^{2 T \lambda} + 1
        \right)
        }{
            2 T^{2} \lambda^{2} - \cosh{\left(2 T \lambda \right)} + 1
        }
```


Damping Rate
```math
    \sigma_\lambda^2 = \frac{\sigma_m^{2}}{N A^{2} }
    \frac{
        -8 T \lambda^{3}  \left(e^{2 T \lambda} - 1\right)
        }{
        2 T^{2} \lambda^{2} - \cosh{\left(2 T \lambda \right)} + 1
    }
```

Angular Frequency
```math
 \sigma_\omega^2 = \frac{\sigma_m^{2}}{N  A^{2}}
    \frac{
        -8 T \lambda^{3}  \left(e^{2 T \lambda} - 1\right)
        }{
             2 T^{2} \lambda^{2} - \cosh{\left(2 T \lambda \right)} + 1
        }
```

Frequency
```math
\sigma_{f}= \frac{\sigma_m}{a}  \frac{\sqrt{2}}{\pi \sqrt{N}}
\sqrt{\frac{  T \lambda_{D}^{3} (-1 +e^{2 T \lambda_{D}} )  }{-1-2 T^2 \lambda_{D}^2 + \cosh{(2 T \lambda_{D})} }}
```

## Comparison With Simultion
The above expressions have excellent agreement to numerical experiments when the observation duration is greater than the oscillation period (here 1 s).
Parameters used in simulation $A=1$ m, $f=1$ Hz, $\lambda_{D}=0.01$, $\sigma=0.01$, $N=10^{3}$.
To reproduce see
```
demos/compare_to_fit.ipynb
```



| ![Uncertainty in Damping Rate](/demos/figures/uncertainty_damping_rate.png "Uncertainty in Damping Rate") |
| :-------------------------------------------------------------------------------------------------------: |
|                                **Figure 2** - Uncertainty in damping rate.                                |


| ![Uncertainty in Frequency](/demos/figures/uncertainty_frequency.png "Uncertainty in Frequency") |
| :----------------------------------------------------------------------------------------------: |
|                             **Figure 3** - Uncertainty in frequency.                             |


| ![Uncertainty in Amplitude](/demos/figures/uncertainty_amplitude.png "Uncertainty in Amplitude") |
| :----------------------------------------------------------------------------------------------: |
|                             **Figure 1** - Uncertainty in amplitude.                             |

| ![Uncertainty in Phase](/demos/figures/uncertainty_phase.png "Uncertainty in Phase") |
| :----------------------------------------------------------------------------------: |
|                         **Figure 4** - Uncertainty in phase.                         |


## Methods
The derivation of the symbolic expresions are performed using a fisher information matrix.
Elements are computed using a integral approximation of sums combined with trigonometric orthogonality conditions.
```
demos/derivation.ipynb
```

## Installation and Usage
We use a development container for reproducible setup so reproducing these results should be a breeze.

### Requirements
- **VS Code**: Install [Visual Studio Code](https://code.visualstudio.com/).
- **Docker**: Install [Docker](https://www.docker.com/)

### Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/brycehenson/fit_error_dampened_sine_wave.git
   cd fit_error_dampened_sine_wave
   ```

2. **Open in VS Code**:
   - Open VS Code.
   - Use the "Open Folder" option to open the cloned repository.

3. **Launch the Dev Container**:
   - Ensure Docker is running.
   - In VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open the Command Palette.
   - Search for and select **"Dev Containers: Reopen in Container"**.

4. **Run Derivations**:
   - Open the notebook at `demos/compare_to_fit.ipynb`. and `demos/derivation.ipynb`.
   - Run the cells to generate derivations,figures and compare numerical fit simulations.

