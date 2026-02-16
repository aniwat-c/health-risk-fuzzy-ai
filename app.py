import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. การตั้งค่าหน้าเว็บและสไตล์ CSS ---
st.set_page_config(page_title="AI Health Advisor Pro", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    /* บังคับ Theme และสีของ Metric */
    .stApp { background-color: #0E1117 !important; }
    [data-testid="stMetricValue"] { color: #000000 !important; }
    [data-testid="stMetricLabel"] { color: #000000 !important; }
    .stMetric {
        background-color: #ffffff !important;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eeeeee;
    }
    h1, h2, h3, h4, label, span { color: #FFFFFF !important; }
    
    /* สไตล์ปุ่ม */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    
    /* กล่องข้อมูลสรุป */
    .summary-box {
        background-color: #1A1C24;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ส่วนของ AI Logic (Fuzzy Logic) ---
# กำหนดช่วงข้อมูล
temp_range = np.arange(34, 43.1, 0.1)
sleep_range = np.arange(0, 13, 1)
stress_range = np.arange(0, 11, 1)
risk_range = np.arange(0, 101, 1)

# สร้างตัวแปร Fuzzy
temp = ctrl.Antecedent(temp_range, 'temperature')
sleep = ctrl.Antecedent(sleep_range, 'sleep_hours')
stress = ctrl.Antecedent(stress_range, 'stress_level')
risk = ctrl.Consequent(risk_range, 'risk')

# Membership Functions
temp['normal'] = fuzz.trimf(temp.universe, [34, 36.5, 37.8])
temp['fever'] = fuzz.trapmf(temp.universe, [37.2, 38.5, 41, 43])
sleep['low'] = fuzz.trimf(sleep.universe, [0, 0, 6])
sleep['normal'] = fuzz.trimf(sleep.universe, [5, 8, 12])
stress['low'] = fuzz.trimf(stress.universe, [0, 0, 5])
stress['high'] = fuzz.trimf(stress.universe, [4, 7, 10])

risk['low'] = fuzz.trimf(risk.universe, [0, 25, 50])
risk['medium'] = fuzz.trimf(risk.universe, [40, 60, 80])
risk['high'] = fuzz.trimf(risk.universe, [70, 85, 100])

# กฎ (Rules)
rules = [
    ctrl.Rule(temp['fever'] & stress['high'], risk['high']),
    ctrl.Rule(temp['normal'] & sleep['normal'] & stress['low'], risk['low']),
    ctrl.Rule(temp['fever'] & sleep['low'], risk['high']),
    ctrl.Rule(temp['normal'] & (sleep['low'] | stress['high']), risk['medium']),
]
health_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# --- 3. ส่วนหน้าจอหลัก ---
st.title("🏥 HealthRisk AI Advisor")
st.markdown("### ระบบประเมินความเสี่ยงสุขภาพเบื้องต้น")
st.divider()

col_input, col_display = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("👤 ข้อมูลพื้นฐาน")
    with st.container(border=True):
        in_sex = st.radio("เพศ", ["ชาย", "หญิง"], horizontal=True)
        in_age = st.number_input("อายุ (ปี)", min_value=1, max_value=120, value=25)
        c_w, c_h = st.columns(2)
        in_weight = c_w.number_input("น้ำหนัก (กก.)", min_value=1.0, value=65.0)
        in_height = c_h.number_input("ส่วนสูง (ซม.)", min_value=50.0, value=170.0)
        
        # คำนวณ BMI อัตโนมัติ
        bmi = in_weight / ((in_height/100)**2)
    
    st.subheader("📋 ระบุตัวบ่งชี้สุขภาพ")
    with st.container(border=True):
        in_temp = st.slider("🌡️ อุณหภูมิร่างกาย (°C)", 35.0, 42.0, 37.0, step=0.1)
        in_sleep = st.select_slider("😴 ชั่วโมงการนอน", options=range(13), value=7)
        in_stress = st.select_slider("🤯 ระดับความเครียด (0-10)", options=range(11), value=3)
    
    btn_calc = st.button("🚀 เริ่มการวิเคราะห์")

with col_display:
    if btn_calc:
        # แสดงกล่องสรุปข้อมูลผู้ใช้
        st.markdown(f"""
        <div class='summary-box'>
            <b>โปรไฟล์ผู้ใช้:</b> เพศ {in_sex} | อายุ {in_age} ปี | <b>BMI: {bmi:.2f}</b>
        </div>
        """, unsafe_allow_html=True)

        # ประมวลผล Fuzzy
        health_sim.input['temperature'] = in_temp
        health_sim.input['sleep_hours'] = in_sleep
        health_sim.input['stress_level'] = in_stress
        
        try:
            health_sim.compute()
            res_risk = health_sim.output['risk']
            
            st.subheader("📊 ผลการวิเคราะห์")
            m1, m2 = st.columns(2)
            status = "ปลอดภัย" if res_risk < 40 else ("ควรระวัง" if res_risk < 70 else "อันตราย")
            m1.metric("ความเสี่ยงสะสม", f"{res_risk:.2f}%")
            m2.metric("สถานะปัจจุบัน", status)

            # แสดงคำแนะนำ
            if res_risk > 70:
                st.error("🚨 **คำแนะนำ:** พบความเสี่ยงสูง ควรพักผ่อนและพบแพทย์ทันที")
            elif res_risk > 40:
                st.warning("⚠️ **คำแนะนำ:** เริ่มมีความเสี่ยง ควรลดความเครียดและนอนหลับให้เพียงพอ")
            else:
                st.success("✅ **คำแนะนำ:** สุขภาพอยู่ในเกณฑ์ดี รักษาสุขอนามัยต่อไป")

            # กราฟ
            st.write("#### 📉 กราฟสรุปผล (Inference Visualization)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(risk_range, risk['low'].mf, 'g', label='Low Risk')
            ax.plot(risk_range, risk['medium'].mf, 'orange', label='Medium Risk')
            ax.plot(risk_range, risk['high'].mf, 'r', label='High Risk')
            ax.axvline(x=res_risk, color='blue', linestyle='--', linewidth=2, label=f'Result ({res_risk:.1f}%)')
            ax.legend()
            st.pyplot(fig)

        except:
            st.error("ไม่สามารถคำนวณได้: ข้อมูลอยู่นอกขอบเขตของกฎที่ตั้งไว้")
    else:
        st.info("กรุณากรอกข้อมูลและกดปุ่ม 'เริ่มการวิเคราะห์'")

# --- 4. ส่วนวิเคราะห์เชิงลึก ---
st.divider()
exp = st.expander("🛠️ ข้อมูลเชิงลึก (Membership Degree Analysis)")
with exp:
    deg_fever = fuzz.interp_membership(temp_range, temp['fever'].mf, in_temp)
    deg_sleep_low = fuzz.interp_membership(sleep_range, sleep['low'].mf, in_sleep)
    deg_stress_high = fuzz.interp_membership(stress_range, stress['high'].mf, in_stress)
    
    st.write(f"ความเป็นสมาชิก 'ไข้': **{deg_fever:.2f}**")
    st.write(f"ความเป็นสมาชิก 'นอนน้อย': **{deg_sleep_low:.2f}**")
    st.write(f"ความเป็นสมาชิก 'เครียดสูง': **{deg_stress_high:.2f}**")
