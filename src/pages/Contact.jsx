import { useState } from 'react'

function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  })

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    alert('Form submitted! (This is a demo - no actual submission)')
    setFormData({ name: '', email: '', message: '' })
  }

  return (
    <div className="page">
      <section className="content">
        <h1>Contact Us</h1>
        <p>Get in touch with us for your next project.</p>
        
        <div className="contact-container">
          <div className="contact-info">
            <h3>Get In Touch</h3>
            <div className="info-item">
              <strong>Email:</strong> hello@nana.com
            </div>
            <div className="info-item">
              <strong>Phone:</strong> +1 (555) 123-4567
            </div>
            <div className="info-item">
              <strong>Address:</strong> 123 Tech Street, Digital City, DC 12345
            </div>
          </div>
          
          <form className="contact-form" onSubmit={handleSubmit}>
            <h3>Send a Message</h3>
            <div className="form-group">
              <input
                type="text"
                name="name"
                placeholder="Your Name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <input
                type="email"
                name="email"
                placeholder="Your Email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <textarea
                name="message"
                placeholder="Your Message"
                rows="5"
                value={formData.message}
                onChange={handleChange}
                required
              ></textarea>
            </div>
            <button type="submit">Send Message</button>
          </form>
        </div>
      </section>
    </div>
  )
}

export default Contact