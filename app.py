import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. การตั้งค่าหน้าเว็บและสไตล์ CSS ---
st.set_page_config(page_title="AI Health Advisor Pro", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    /* บังคับเฉพาะ Metric ให้เป็นสีดำบนพื้นขาวเสมอ */
    [data-testid="stMetricValue"] { color: #000000 !important; }
    [data-testid="stMetricLabel"] { color: #000000 !important; }
    .stMetric {
        background-color: #ffffff !important;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eeeeee;
    }
    
    h1, h2, h3, h4 { color: #FFFFFF !important; }
    
    .stButton>button {
        background-color: #007bff;
        color: white;
        width: 100%;
        border-radius: 10px;
    }

    /* สไตล์สำหรับกล่องข้อความอธิบายข้อมูลส่วนบุคคล */
    .personal-info {
        background-color: #1A1C24;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ส่วนของ AI Logic (Fuzzy Logic) ---
temp_range = np.arange(34, 43.1, 0.1)
sleep_range = np.arange(0, 13, 1)
stress_range = np.arange(0, 11, 1)
risk_range = np.arange(0, 101, 1)

temp = ctrl.Antecedent(temp_range, 'temperature')
sleep = ctrl.Antecedent(sleep_range, 'sleep_hours')
stress = ctrl.Antecedent(stress_range, 'stress_level')
risk = ctrl.Consequent(risk_range, 'risk')

temp['normal'] = fuzz.trimf(temp.universe, [34, 36.5, 37.8])
temp['fever'] = fuzz.trapmf(temp.universe, [37.2, 38.5, 41, 43])
sleep['low'] = fuzz.trimf(sleep.universe, [0, 0, 6])
sleep['normal'] = fuzz.trimf(sleep.universe, [5, 8, 12])
stress['low'] = fuzz.trimf(stress.universe, [0, 0, 5])
stress['high'] = fuzz.trimf(stress.universe, [4, 7, 10])

risk['low'] = fuzz.trimf(risk.universe, [0, 25, 50])
risk['medium'] = fuzz.trimf(risk.universe, [40, 60, 80])
risk['high'] = fuzz.trimf(risk.universe, [70, 85, 100])

rules = [
    ctrl.Rule(temp['fever'] & stress['high'], risk['high']),
    ctrl.Rule(temp['normal'] & sleep['normal'] & stress['low'], risk['low']),
    ctrl.Rule(temp['fever'] & sleep['low'], risk['high']),
    ctrl.Rule(temp['normal'] & (sleep['low'] | stress['high']), risk['medium']),
]
health_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# --- 3. ส่วนหน้าจอหลัก ---
st.title("🏥 HealthRisk AI Advisor")
st.markdown("### ระบบประเมินความเสี่ยงสุขภาพด้วย *Mamdani Fuzzy Inference System*")
st.divider()

col_input, col_display = st.columns([1, 2], gap="large")

with col_input:
    # --- ส่วนที่เพิ่มใหม่: ข้อมูลส่วนบุคคล ---
    st.subheader("👤 ข้อมูลทั่วไป")
    with st.container(border=True):
        c_sex, c_age = st.columns(2)
        in_sex = c_sex.selectbox("เพศ", ["ชาย", "หญิง", "ไม่ระบุ"])
        in_age = c_age.number_input("อายุ (ปี)", min_value=1, max_value=120, value=25)
        
        c_weight, c_height = st.columns(2)
        in_weight = c_weight.number_input("น้ำหนัก (กก.)", min_value=1.0, value=60.0)
        in_height = c_height.number_input("ส่วนสูง (ซม.)", min_value=50.0, value=170.0)
        
        # คำนวณ BMI เบื้องต้น
        bmi = in_weight / ((in_height/100)**2)
    
    st.subheader("📋 ระบุข้อมูลตัวบ่งชี้")
    with st.container(border=True):
        in_temp = st.slider("🌡️ อุณหภูมิร่างกาย (°C)", 35.0, 42.0, 37.0, step=0.1)
        in_sleep = st.select_slider("😴 ชั่วโมงการนอนต่อวัน", options=range(13), value=7)
        in_stress = st.select_slider("🤯 ระดับความเครียด (0-10)", options=range(11), value=3)
    
    btn_calc = st.button("🚀 วิเคราะห์ผลลัพธ์")

with col_display:
    if btn_calc:
        # แสดงข้อมูลส่วนบุคคลที่สรุปแล้ว
        st.markdown(f"""
        <div class='personal-info'>
            <b>สรุปข้อมูลส่วนบุคคล:</b> เพศ {in_sex} | อายุ {in_age} ปี | BMI: {bmi:.1f}
        </div>
        """, unsafe_allow_html=True)

        health_sim.input['temperature'] = in_temp
        health_sim.input['sleep_hours'] = in_sleep
        health_sim.input['stress_level'] = in_stress
        
        try:
            health_sim.compute()
            res_risk = health_sim.output['risk']
            
            st.subheader("📊 ผลการวิเคราะห์")
            m1, m2 = st.columns(2)
            status = "ปลอดภัย" if res_risk < 40 else ("ควรระวัง" if res_risk < 70 else "อันตราย")
            m1.metric("ความเสี่ยงโดยรวม", f"{res_risk:.2f}%")
            m2.metric("สถานะสุขภาพ", status)

            st.write("##### **คำแนะนำจากระบบ:**")
            if res_risk > 70:
                st.error("🚨 **สถานะ: อันตราย**")
                st.markdown("* ดื่มน้ำมากๆ และทานยาลดไข้\n* ควรหาเวลางีบพักผ่อนให้เพียงพอ\n* **โปรดพบแพทย์ทันที** หากอาการไม่ดีขึ้น")
            elif res_risk > 40:
                st.warning("⚠️ **สถานะ: ควรระวัง**")
                st.markdown("* ควรลดภาระงานเพื่อลดความเครียดสะสม\n* พยายามนอนหลับให้ครบ 7-8 ชั่วโมง\n* สังเกตอาการอย่างใกล้ชิด")
            else:
                st.success("✅ **สถานะ: ปลอดภัย**")
                st.markdown("* รักษาสุขภาพและพฤติกรรมที่ดีต่อไป\n* ออกกำลังกายอย่างสม่ำเสมอ")

            # กราฟ
            st.write("#### 📉 กราฟสรุปผล (Inference Visualization)")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(risk_range, risk['low'].mf, 'g', label='Low')
            ax.plot(risk_range, risk['medium'].mf, 'y', label='Medium')
            ax.plot(risk_range, risk['high'].mf, 'r', label='High')
            ax.axvline(x=res_risk, color='blue', linestyle='--', linewidth=2, label=f'Result ({res_risk:.1f}%)')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error("ไม่สามารถคำนวณได้: ข้อมูลไม่เข้าเงื่อนไขของกฎ")
    else:
        st.info("กรุณาป้อนข้อมูลเพื่อเริ่มการวิเคราะห์")

# --- 4. ส่วนวิเคราะห์เชิงลึก ---
st.divider()
exp = st.expander("🛠️ ดูเบื้องหลังการทำงาน (Membership Degree Analysis)")
with exp:
    st.info("ค่าความเป็นสมาชิกตามหลักการ Fuzzy Logic")
    c1, c2, c3 = st.columns(3)
    deg_fever = fuzz.interp_membership(temp_range, temp['fever'].mf, in_temp)
    deg_sleep_low = fuzz.interp_membership(sleep_range, sleep['low'].mf, in_sleep)
    deg_stress_high = fuzz.interp_membership(stress_range, stress['high'].mf, in_stress)
    
    c1.write(f"ความเป็นสมาชิก 'ไข้': **{deg_fever:.2f}**")
    c2.write(f"ความเป็นสมาชิก 'นอนน้อย': **{deg_sleep_low:.2f}**")
    c3.write(f"ความเป็นสมาชิก 'เครียดสูง': **{deg_stress_high:.2f}**")
