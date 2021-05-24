# RowSPCA-CS282-Machine-Learning-Course-Project
- Reproduction of *Upper bounds for Model-Free Row-Sparse Principal Component Analysis*. 
- Course project for CS282, Machine Learning, SIST, ShanghaiTech University in 2021 Spring.
- Authored by **Ziyi Yu**, **Yichen Zhu** & **Tao Huang**.

# Requirements
- [numpy](https://github.com/numpy/numpy)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [mosek](https://www.mosek.com/)(Use `pip` to install)
- [gurobipy](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_using_pip_to_install_gr.html)(Use `pip` to install)

# Table Of Contents
- [RowSPCA-CS282-Machine-Learning-Course-Project](#rowspca-cs282-machine-learning-course-project)
- [Requirements](#requirements)
- [Table Of Contents](#table-of-contents)
- [How to use](#how-to-use)
- [In Details](#in-details)
- [Contact Us](#contact-us)

# How to use   
You can find a simple example in `main.py`.

# In Details
```
├──  chat
│    └── chat.md  - email chat with author.
│
│
├──  code  
│    └── baseline1.py    - The baseline 1 in the paper.
│    └── baseline2.py    - The baseline 2 in the paper.
│    └── localSearch.py  - The local search algorithm to find a lower (primal) bound.
│    └── main.py         - A simple example.
│    └── PLA_SOCP.py     - Solve the SOCP via PLA technique. Run in gurobipy. 
│    └── Plot.mlx        - Plot the figs in our report. Datas from result.xlsx
│ 
│
├──  dataset  
│    └── Reddit  - Reddit dataset.
│    └── Others  - Other dataset.
│
│
├──  paper
│    └── wang20e.pdf     - A modified paper of the conference version.
│
│
└──  result              - auxiliary files for plotting figures.

```


# Contact Us
You are welcome to contact us for any problem.
- Ziyi Yu `yuzy@shanghaitech.edu.cn`
- Yichen Zhu `zhuych@shanghaitech.edu.cn`
- Huang Tao `huangtao1@shanghaitech.edu.cn`