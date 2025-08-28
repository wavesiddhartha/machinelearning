function Services() {
  const services = [
    {
      title: "Web Development",
      description: "Custom web applications built with modern technologies",
      icon: "🌐"
    },
    {
      title: "Mobile Development", 
      description: "Cross-platform mobile apps using React Native",
      icon: "📱"
    },
    {
      title: "UI/UX Design",
      description: "Beautiful and intuitive user interfaces and experiences",
      icon: "🎨"
    },
    {
      title: "Consulting",
      description: "Technical consulting and architecture guidance",
      icon: "💡"
    }
  ]

  return (
    <div className="page">
      <section className="content">
        <h1>Our Services</h1>
        <p>We offer a range of digital services to help your business grow.</p>
        
        <div className="services-grid">
          {services.map((service, index) => (
            <div key={index} className="service-card">
              <div className="service-icon">{service.icon}</div>
              <h3>{service.title}</h3>
              <p>{service.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default Services