# Author: Kaituo Xu, Fan Yu
import numpy as np

def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    '''
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    '''
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    ai = [[0] * N for _ in range(T)]
    # Begin Assignment
    for t in range(T):
        if t == 0:
            for i in range(N):
                # 0时刻，发射概率*观测概率
                ai[t][i] = pi[i] * B[i][O[t]]
        else:
            for i in range(N):
                trans_p = 0
                for j in range(N):
                    # t-1为j * j->i状态转移
                    trans_p += ai[t-1][j] * A[j][i]
                # t时刻为i = trans_p * t观测为i
                ai[t][i] = trans_p * B[i][O[t]]
    # Put Your Code Here
    prob = sum(ai[T-1])
    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment
    Bi = [[0] * N for _ in range(T)]
    # 初始化
    for i in range(N):
        Bi[T-1][i] = 1
    # 向前遍历
    for t in range(T-2, -1, -1):
        for i in range(N):
            trans = 0
            for j in range(N):
                # i->j转移矩阵 * t+1为j观测概率 * t+1时刻j概率
                trans += A[i][j] * B[j][O[t+1]] * Bi[t+1][j]
            # t时刻i概率
            Bi[t][i] = trans    
    # Put Your Code Here
    for i in range(N):
        # 发射矩阵*观测矩阵*0时刻i概率
        prob += Bi[0][i] * pi[i] * B[i][O[0]]
    # End Assignment
    return prob
 

def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment
    score = [[0] * N for _ in range(T)]
    link = [[0] * N for _ in range(T)]
    ai = [[0] * N for _ in range(T)]
    for t in range(T):
        if t == 0:
            for i in range(N):
                ai[t][i] = pi[i] * B[i][O[t]]
        else:
            for i in range(N):
                for j in range(N):
                    # 选出最大的概率路径 N*N
                    if score[t][i] < ai[t-1][j] * A[j][i]:
                        score[t][i] = ai[t-1][j] * A[j][i]
                        # 记录前身节点
                        link[t][j] = i
                    # 最后概率分数
                    ai[t][i] = score[t][i] * B[i][O[t]]

    best_prob = max(ai[T-1])
    max_sum = 0

    for i in range(N):
        if max_sum < score[T-1][i]:
            last = i
    best_path.append(last)
    
    for t in range(T-1, 0, -1):
        last = link[t][last]
        best_path.append(last)
    best_path = list(reversed(best_path))
    
    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model) 
    print(best_prob, best_path)
