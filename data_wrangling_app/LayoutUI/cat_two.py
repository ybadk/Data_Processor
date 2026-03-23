import streamlit as st

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    <article class="card">
  <div class="card-img">
    <div class="card-imgs pv delete"></div>
  </div>

  <div class="project-info">
    <div class="flex">
      <div class="project-title">Title card</div>
      <span class="tag">type</span>
    </div>
    <span class="lighter"
      >Lorem ipsum, dolor sit amet consectetur adipisicing elit. Repudiandae
      voluptas ullam aut incidunt minima.</span
    >
  </div>
</article>

<style>
/* From Uiverse.io by Kagamiie  - Tags: simple, card, about me, profilecard */
.project-info {
  padding: 100px 40px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  position: relative;
  top: -50px;
}

.project-title {
  font-weight: 500;
  font-size: 1.5em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: black;
}

.lighter {
  font-size: 0.9em;
}

.flex {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tag {
  font-weight: lighter;
  color: grey;
}

/*DELETE THIS TWO LINE*/
.delete {
  background-color: #b2b2fd;
}

.card-img div {
  width: 90%;
}
/*IF USING IMAGES*/

.card {
  background-color: white;
  color: black;
  width: 300px;
  max-height: 330px;
  border-radius: 8px;
  box-shadow: rgba(50, 50, 93, 0.25) 0px 50px 100px -20px,
    rgba(0, 0, 0, 0.3) 0px 30px 60px -30px;
}

.card-img {
  position: relative;
  top: -20px;
  height: 100px;
  display: flex;
  justify-content: center;
}

/* Change the .card-img div to .card-img img to use img*/
.card-img a,
.card-img div {
  height: 150px;
  width: 90%;
  /* Change this width here to change the width of the color/image */
  object-fit: cover;
  border-radius: 8px;
  box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
}

.card-imgs {
  transition: all 0.5s;
}

</style>
    
    """,
        unsafe_allow_html=True,
    )


with col2:
    st.markdown("""
    
    """, unsafe_allow_html=True)

st.divider()

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("")
with col4:
    st.markdown("")
with col5:
    st.markdown(
        """
    <article class="card">
  <div class="card-img">
    <div class="card-imgs pv delete"></div>
  </div>

  <div class="project-info">
    <div class="flex">
      <div class="project-title">Title card</div>
      <span class="tag">type</span>
    </div>
    <span class="lighter"
      >Lorem ipsum, dolor sit amet consectetur adipisicing elit. Repudiandae
      voluptas ullam aut incidunt minima.</span
    >
  </div>
</article>

<style>
/* From Uiverse.io by Kagamiie  - Tags: simple, card, about me, profilecard */
.project-info {
  padding: 100px 40px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  position: relative;
  top: -50px;
}

.project-title {
  font-weight: 500;
  font-size: 1.5em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: black;
}

.lighter {
  font-size: 0.9em;
}

.flex {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tag {
  font-weight: lighter;
  color: grey;
}

/*DELETE THIS TWO LINE*/
.delete {
  background-color: #b2b2fd;
}

.card-img div {
  width: 90%;
}
/*IF USING IMAGES*/

.card {
  background-color: white;
  color: black;
  width: 300px;
  max-height: 330px;
  border-radius: 8px;
  box-shadow: rgba(50, 50, 93, 0.25) 0px 50px 100px -20px,
    rgba(0, 0, 0, 0.3) 0px 30px 60px -30px;
}

.card-img {
  position: relative;
  top: -20px;
  height: 100px;
  display: flex;
  justify-content: center;
}

/* Change the .card-img div to .card-img img to use img*/
.card-img a,
.card-img div {
  height: 150px;
  width: 90%;
  /* Change this width here to change the width of the color/image */
  object-fit: cover;
  border-radius: 8px;
  box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
}

.card-imgs {
  transition: all 0.5s;
}

</style>
    
    """,
        unsafe_allow_html=True,
    )
