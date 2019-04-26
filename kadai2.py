import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate, integrate

File = "CV_MOSCap-#001.dat"

###物理定数###
Kox = 3.9
Ksi = 11.9
Epsilon0 = 8.85*1e-14 #F/cm
Area = 1.0*1e-12 #m^2
Kb = 1.38*1e-23 #J/K
Q = 1.60*1e-19 #C
T = 300 #K
Ni = 1.45*1e10 #cm^-3


def main():
    ###グローバル変数宣言###
    global GateVoltage, CapacitancePerArea

    ###データ取り込み###
    Data = np.loadtxt(File)
    GateVoltage = Data[:,0] #V
    Capacitance = Data[:,1] #F
    CapacitancePerArea = Capacitance/Area/1e+4 #F/cm^2

    ###(1)###
    MaxCapacitancePerArea = np.max(CapacitancePerArea) #F/cm^2
    Tox = Kox*Epsilon0/MaxCapacitancePerArea #cm
    print ("酸化膜厚は{0:1.5e}cm".format(Tox))

    MinCapacitancePerArea = np.min(CapacitancePerArea) #F/cm^2
    Cdm = (1/MinCapacitancePerArea - 1/MaxCapacitancePerArea)**-1 #F/cm^2
    Xdm = Ksi*Epsilon0/Cdm
    print ("最大空乏層暑さ{0:1.5e}cm".format(Xdm))

    ###(2)###
    ###まずは基板濃度を絞り込む###
    MinNsub = 1.5e+10
    MaxNsub = 1.0e+20
    ###誤差を設定###
    Error = 1.0e+10

    ###2分法を実行する###
    while True:
        if abs(MinNsub-MaxNsub) > Error:
            Mid = (MinNsub+MaxNsub)/2

            if non_linear_equations(Mid,Xdm) < 0:
                MinNsub = Mid
            else:
                MaxNsub = Mid
        else:
            break

    ###中間値を答えとする###
    Nsub = Mid #cm^-3
    print ("基板不純物濃度は{0:1.5e}".format(Nsub))


    ###(3)###
    CsFB = math.sqrt(Ksi*Epsilon0*Q**2*Nsub/Kb/T) # F/cm^2
    CFB = (1/CsFB+1/MaxCapacitancePerArea)**-1

    ###線形近似，3次スプライン補間(のちの計算のためx軸y軸を反転させておく)###
    spline_interpolation_1d = interpolate.interp1d(CapacitancePerArea, GateVoltage, kind="slinear")
    spline_interpolation_2d = interpolate.interp1d(CapacitancePerArea, GateVoltage, kind="quadratic")
    spline_interpolation_3d = interpolate.interp1d(CapacitancePerArea, GateVoltage, kind="cubic")


    ###CFBの時のゲート電圧(フラットバンド電圧)を求める###
    VFB1d = spline_interpolation_1d(CFB)
    VFB2d = spline_interpolation_2d(CFB)
    VFB3d = spline_interpolation_3d(CFB)

    print ("線形近似ではフラットバンド電圧は{0:1.5f}Vである".format(VFB1d))
    print ("2次スプライン補間ではフラットバンド電圧は{0:1.5f}Vである".format(VFB2d))
    print ("3次スプライン補間ではフラットバンド電圧は{0:1.5f}Vである".format(VFB3d))

    ###おまけ####
    ###CVグラフ描画###
    ###生データ###
    ###図1###
    plt.subplot(2, 1, 1)
    plt.title("CV-graph")
    CVgraph = plt.plot(GateVoltage, CapacitancePerArea, marker="o", ls='None')
    MaxGateVoltage = np.max(GateVoltage)
    MinGateVoltage = np.min(GateVoltage)
    ###CFBの位置に横線を描画###
    CVgraph = plt.hlines(CFB, MinGateVoltage, MaxGateVoltage, color='pink', label="CFB={0:.3e}".format(CFB))
    ###VFB(線形近似)の位置に縦線を描画###
    CVgraph = plt.vlines(VFB1d, MinCapacitancePerArea, MaxCapacitancePerArea, color='blue', label="VFB={0:.3f}".format(VFB1d))

    ###スプライン補間描画用のデータセット作成###
    VirtualCapacitancePerArea = np.linspace(MinCapacitancePerArea, MaxCapacitancePerArea, 50)

    ###スプライン補間関数を描画###
    plt.plot(spline_interpolation_1d(VirtualCapacitancePerArea), VirtualCapacitancePerArea,label="1d")
    #plt.plot(spline_interpolation_2d(VirtualCapacitancePerArea), VirtualCapacitancePerArea,label="2d")
    #plt.plot(spline_interpolation_3d(VirtualCapacitancePerArea), VirtualCapacitancePerArea, label="3d")
    ###凡例表示###
    plt.legend()

    ###おまけ終わり###

    ###(4)###
    ###ゲート電圧の点数を抽出###
    PointVg = GateVoltage.shape[0]
    ###データを入れる箱用意．0行で1行の要素数が2###
    Data = np.zeros((0,2))
    ###積分###
    for i in range(PointVg):
        x = np.linspace(0, GateVoltage[i], num=45)
        y = (cal_gate_capacitance(x)-MinCapacitancePerArea)/Q
        Ns = abs(integrate.simps(x,y))
        Data = np.append(Data, [[GateVoltage[i], Ns]], axis=0)

    ###データ保存###
    np.savetxt('Ns-Vg.dat', Data)

    ###グラフ表示###
    plt.subplot(2,1,2)
    plt.plot(Data[:,0], Data[:,1])
    ###図を表示###
    plt.show()
    return 0

###非線形方程式を定義する###
def non_linear_equations(Nsub,Xdm):
    Value = Xdm-math.sqrt(4*Ksi*Epsilon0*Kb*T*math.log(Nsub/Ni)/Q/Q/Nsub)
    return Value

###積分用の関数###
def cal_gate_capacitance(Vg):
    spline_interpolation_1d_x_vg = interpolate.interp1d(GateVoltage, CapacitancePerArea, kind="slinear")
    Value = spline_interpolation_1d_x_vg(Vg)
    return Value


if __name__ == '__main__':
    main()
