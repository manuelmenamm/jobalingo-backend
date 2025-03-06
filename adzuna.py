import os
import requests
import random
import re
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

class IndustryClassifier:
    """Industry classifier using cosine similarity with Spanish templates"""
    
    def __init__(self):
        # Spanish stop words list
        self.spanish_stop_words = [
            'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
            'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella',
            'ellas', 'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres', 'es',
            'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estabais', 'estaban',
            'estabas', 'estad', 'estada', 'estadas', 'estado', 'estados', 'estamos', 'estando',
            'estar', 'estaremos', 'estará', 'estarán', 'estarás', 'estaré', 'estaréis',
            'estaría', 'estaríais', 'estaríamos', 'estarían', 'estarías', 'estas', 'este',
            'estemos', 'esto', 'estos', 'estoy', 'estuve', 'estuviera', 'estuvierais',
            'estuvieran', 'estuvieras', 'estuvieron', 'estuviese', 'estuvieseis', 'estuviesen',
            'estuvieses', 'estuvimos', 'estuviste', 'estuvisteis', 'estuviéramos',
            'estuviésemos', 'estuvo', 'está', 'estábamos', 'estáis', 'están', 'estás', 'esté',
            'estéis', 'estén', 'estés', 'fue', 'fuera', 'fuerais', 'fueran', 'fueras', 'fueron',
            'fuese', 'fueseis', 'fuesen', 'fueses', 'fui', 'fuimos', 'fuiste', 'fuisteis',
            'fuéramos', 'fuésemos', 'ha', 'habida', 'habidas', 'habido', 'habidos', 'habiendo',
            'habremos', 'habrá', 'habrán', 'habrás', 'habré', 'habréis', 'habría', 'habríais',
            'habríamos', 'habrían', 'habrías', 'habéis', 'había', 'habíais', 'habíamos',
            'habían', 'habías', 'han', 'has', 'hasta', 'hay', 'haya', 'hayamos', 'hayan',
            'hayas', 'hayáis', 'he', 'hemos', 'hube', 'hubiera', 'hubierais', 'hubieran',
            'hubieras', 'hubieron', 'hubiese', 'hubieseis', 'hubiesen', 'hubieses', 'hubimos',
            'hubiste', 'hubisteis', 'hubiéramos', 'hubiésemos', 'hubo', 'la', 'las', 'le',
            'les', 'lo', 'los', 'me', 'mi', 'mis', 'mucho', 'muchos', 'muy', 'más', 'mí',
            'mía', 'mías', 'mío', 'míos', 'nada', 'ni', 'no', 'nos', 'nosotras', 'nosotros',
            'nuestra', 'nuestras', 'nuestro', 'nuestros', 'o', 'os', 'otra', 'otras', 'otro',
            'otros', 'para', 'pero', 'poco', 'por', 'porque', 'que', 'quien', 'quienes', 'qué',
            'se', 'sea', 'seamos', 'sean', 'seas', 'seremos', 'será', 'serán', 'serás', 'seré',
            'seréis', 'sería', 'seríais', 'seríamos', 'serían', 'serías', 'seáis', 'sido',
            'siendo', 'sin', 'sobre', 'sois', 'somos', 'son', 'soy', 'su', 'sus', 'suya',
            'suyas', 'suyo', 'suyos', 'sí', 'también', 'tanto', 'te', 'tendremos', 'tendrá',
            'tendrán', 'tendrás', 'tendré', 'tendréis', 'tendría', 'tendríais', 'tendríamos',
            'tendrían', 'tendrías', 'tened', 'tenemos', 'tenga', 'tengamos', 'tengan', 'tengas',
            'tengo', 'tengáis', 'tenida', 'tenidas', 'tenido', 'tenidos', 'teniendo', 'tenéis',
            'tenía', 'teníais', 'teníamos', 'tenían', 'tenías', 'ti', 'tiene', 'tienen',
            'tienes', 'todo', 'todos', 'tu', 'tus', 'tuve', 'tuviera', 'tuvierais', 'tuvieran',
            'tuvieras', 'tuvieron', 'tuviese', 'tuvieseis', 'tuviesen', 'tuvieses', 'tuvimos',
            'tuviste', 'tuvisteis', 'tuviéramos', 'tuviésemos', 'tuvo', 'tuya', 'tuyas', 'tuyo',
            'tuyos', 'tú', 'un', 'una', 'uno', 'unos', 'vosotras', 'vosotros', 'vuestra',
            'vuestras', 'vuestro', 'vuestros', 'y', 'ya', 'yo', 'él', 'éramos'
        ]
        
        # Define industry templates in Spanish
        self.industry_templates = {
            'Tecnología': """
                desarrollador software programación programador web aplicaciones móviles frontend backend fullstack
                java javascript python php typescript react angular vue node.js django flask
                devops cloud aws azure google arquitecto software ingeniero informático sistemas
                seguridad ciberseguridad datos big data machine learning inteligencia artificial
                analista programador desarrollo apps tech it tecnología software scrum agile
                base de datos sql nosql mongodb postgresql mysql oracle tecnologías desarrollo
                api rest microservicios docker kubernetes git github gitlab cicd integración
                aplicación web móvil android ios testing qa automatización calidad software
                desarrollador senior programador junior informática tecnológica sistemas
            """,
            
            'Finanzas': """
                finanzas contabilidad financiero contable banca inversión auditoría auditor
                controller análisis financiero analista finanzas fiscal impuestos tributación
                gestor patrimonios presupuesto tesorería riesgo financiero economista economía
                banking investment fund manager portfolio mercados financieros bolsa valores
                contable financiero administrativo cuentas pagar cobrar facturación nóminas
                conciliación bancaria cierre contable balance estados financieros pymes reporting
                consolidación presupuestación previsión forecasting controlling kpi finanzas
                fiscalidad tributación irpf iva impuesto sociedades planificación fiscal pymes
                banca privada gestión patrimonios inversiones asesoramiento financiero wealth
            """,
            
            'Consultoría': """
                consultoría consultor asesoramiento estrategia negocio transformación gestión cambio
                business intelligence mejora procesos optimización eficiencia operativa consultor
                estratégico análisis negocio consultor management strategy advisory consulting 
                consultor senior manager partner director consultoría estratégica operativa 
                procesos erp sap implementación proyectos consultoría empresarial servicios profesionales
                management consulting advisory business strategy change management roadmap
                diagnóstico soluciones empresariales mejora rendimiento transformación digital
                optimización procesos reingeniería gestión proyectos metodologías análisis datos
                cuadro mando kpi indicadores balanced scorecard gestión clientes stakeholders
            """,
            
            'Marketing': """
                marketing publicidad comunicación marca branding estrategia digital redes sociales
                seo sem posicionamiento campañas publicitarias contenidos marketing online 
                marketing digital community manager social media facebook instagram twitter
                analítica web google analytics conversión inbound marketing lead generation
                copywriting email marketing automation crm gestión relaciones cliente experiencia usuario
                estrategia de contenidos blog newsletter campaign manager planificación medios
                publicidad digital programática display native advertising retargeting
                marketing performance móvil app store optimization growth hacking marketing
                dirección marketing product marketing lanzamiento producto market research 
            """,
            
            'Salud': """
                médico enfermero fisioterapeuta farmacéutico odontólogo psicólogo nutricionista
                técnico laboratorio radiología enfermería auxiliar clínica salud sanitario
                hospital clínica centro médico salud pública prevención riesgos laborales
                farmacia industria farmacéutica ensayos clínicos investigación médica
                atención primaria especialista medicina interna cirugía oncología pediatría
                geriatría traumatología cardiología neurología psiquiatría radiología
                laboratorio análisis clínicos diagnóstico tratamiento pacientes cuidados
                rehabilitación terapia ocupacional logopedia óptica optometría podología
                veterinaria auxiliar técnico cuidados auxiliares enfermería tcae due ats
            """,
            
            'Educación': """
                profesor maestro docente enseñanza educación formación pedagógico didáctico
                colegio escuela instituto universidad centro educativo educación infantil
                primaria secundaria bachillerato formación profesional educación especial
                orientador pedagogo educador social tutor formador academia idiomas lenguas
                educación adultos educación continua taller curso formación online elearning
                contenidos educativos material didáctico currículum educativo programación
                didáctica evaluación aprendizaje competencias educativas departamento
                educativo dirección académica coordinación pedagógica educación superior
                investigación educativa innovación docente metodologías activas cooperativo
            """,
            
            'Ingeniería': """
                ingeniero ingeniería civil industrial mecánica eléctrica electrónica telecomunicaciones
                proyectos infraestructura construcción obra arquitecto técnico aparejador
                edificación topógrafo delineante cad diseño industrial producción fabricación
                automatización robótica procesos lean manufacturing calidad ingeniería
                energías renovables eficiencia energética sostenibilidad medioambiente
                instalaciones mantenimiento industrial maquinaria equipos infraestructuras
                obra civil estructuras instalaciones eléctricas climatización fontanería
                telecomunicaciones redes cálculo estructural arquitectura técnica edificación
                dirección obra project manager jefe proyecto ingeniería desarrollo producto
            """,
            
            'Diseño': """
                diseñador gráfico web ux ui producto industrial moda editorial arte artista
                creativo dirección arte ilustrador diseño packaging multimedia animación
                motion graphics modelado 3d diseño interior espacios gráfica publicitaria
                identidad corporativa imagen marca tipografía color composición maquetación
                editorial arte final preimpresión imprenta diseño digital web responsive
                prototipado wireframes usabilidad experiencia usuario interfaces sketch
                photoshop illustrator indesign after effects premiere figma adobe creative
                suite diseño productos industriales muebles objetos diseño moda textil
                patronaje confección diseño interiores decoración escaparatismo retail
            """,
            
            'Ventas': """
                comercial ventas account manager key account vendedor asesor comercial negocio
                cliente grandes cuentas retail distribución canal exportación internacional
                business development negociación cierre ventas preventa posventa comercio
                relación cliente gestión cartera clientes fidelización captación prospección
                territorio zona manager director comercial jefe ventas delegado comercial
                representante agente comercial autónomo comisión incentivos objetivos ventas
                sector estrategia comercial desarrollo negocio nuevos mercados expansión
                distribución canal mayorista minorista retail franquicia exportación import
                comercio internacional gestión punto venta tienda responsable establecimiento
            """,
            
            'Recursos Humanos': """
                recursos humanos rrhh gestión personas talento selección reclutamiento headhunting
                entrevista candidatos assessment center desarrollo profesional formación
                capacitación onboarding plan carrera evaluación desempeño competencias
                retribución compensación beneficios nóminas administración personal relaciones
                laborales convenio negociación sindical prevención riesgos laborales seguridad
                salud laboral clima organización cultura empresarial employer branding atracción
                talento retención plan sucesión gestión conocimiento hr business partner hrbp
                director rrhh responsable técnico generalista especialista selección formación
                administración personal organización empresarial desarrollo organizacional
            """
        }
        
        # Initialize TF-IDF vectorizer with custom Spanish stop words
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=1,            # Include terms that appear at least once
            max_df=0.95,         # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True,   # Apply sublinear tf scaling
            stop_words=self.spanish_stop_words  # Use custom Spanish stop words list
        )
        
        # Fit and transform the templates
        self.industry_names = list(self.industry_templates.keys())
        templates_text = [self.industry_templates[industry] for industry in self.industry_names]
        self.industry_vectors = self.vectorizer.fit_transform(templates_text)
    
    def predict_industry(self, text: str) -> Tuple[str, float]:
        """
        Predict the industry of a job based on its description and title
        Returns the predicted industry and its similarity score
        """
        # Clean the text and ensure it's in lowercase
        text = text.lower()
        
        # Transform the job text
        job_vector = self.vectorizer.transform([text])
        
        # Calculate cosine similarity with each industry template
        similarities = cosine_similarity(job_vector, self.industry_vectors)[0]
        
        # Find the industry with highest similarity
        max_idx = np.argmax(similarities)
        industry = self.industry_names[max_idx]
        similarity = similarities[max_idx]
        
        return industry, similarity


class AdzunaJobParser:
    def __init__(self):
        self.base_url = 'http://api.adzuna.com/v1/api/jobs'
        self.app_id = os.getenv("ADZUNA_APP_ID")
        self.app_key = os.getenv("ADZUNA_APP_KEY")
        self.country = 'es'  # Spain
        
        # Map our frontend industry names to Adzuna category tags
        self.industry_to_category = {
            'Tecnología': ['it-jobs', 'tech-jobs', 'scientific-jobs', 'engineering-jobs'],
            'Finanzas': ['accounting-finance-jobs', 'banking-jobs'],
            'Consultoría': ['consultancy-jobs'],
            'Marketing': ['marketing-jobs', 'pr-advertising-marketing-jobs'],
            'Salud': ['healthcare-nursing-jobs', 'social-work-jobs', 'medical-pharmaceutical-jobs'],
            'Educación': ['teaching-jobs', 'education-jobs'],
            'Ingeniería': ['engineering-jobs', 'energy-jobs'],
            'Diseño': ['creative-design-jobs'],
            'Ventas': ['sales-jobs', 'retail-jobs'],
            'Recursos Humanos': ['hr-jobs']
        }
        
        # Reverse mapping from Adzuna categories to our industry names
        self.category_to_industry = {}
        for industry, categories in self.industry_to_category.items():
            for category in categories:
                self.category_to_industry[category] = industry
        
        # Initialize the industry classifier
        self.industry_classifier = IndustryClassifier()

    # Rest of the code remains the same...

    def fetch_jobs(self, params: Dict = None) -> List[Dict]:
        default_params = {
            'app_id': self.app_id,
            'app_key': self.app_key,
            'results_per_page': 50,
        }
        
        # Merge default and provided params
        request_params = {**default_params}
        
        # Handle location filter
        if params and 'location' in params and params['location'] and len(params['location']) > 0:
            request_params['where'] = params['location'][0]
        else:
            request_params['where'] = 'madrid'  # Default location
        
        # Handle query terms
        if params and 'query' in params and params['query']:
            request_params['what'] = params['query']
        
        # Handle industry filter by mapping to Adzuna categories
        if params and 'industry' in params and params['industry'] and len(params['industry']) > 0:
            # Get the first industry (we'll do further filtering for multiple industries later)
            industry = params['industry'][0]
            if industry in self.industry_to_category:
                # Use the first category for this industry in the API request
                request_params['category'] = self.industry_to_category[industry][0]
        
        try:
            print("Request Parameters:", request_params)
            
            response = requests.get(f'{self.base_url}/{self.country}/search/1', params=request_params)
            print("Response Status Code:", response.status_code)
            
            # Ensure successful response
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            print(f"Total jobs retrieved: {len(data.get('results', []))}")
            
            transformed_jobs = []
            for job in data.get('results', []):
                # Get the job category from Adzuna
                adzuna_category = job.get('category', {}).get('tag', '')
                
                # Map Adzuna category to our industry format
                industry = self._determine_industry(job, adzuna_category)
                
                transformed_job = {
                    'id': str(job.get('id', '')),
                    'title': job.get('title', ''),
                    'company': job.get('company', {}).get('display_name', ''),
                    'location': self._format_location(job),
                    'description': job.get('description', ''),
                    'salary': self._format_salary(job),
                    'jobType': self._determine_job_type(job),
                    'industry': industry,
                    'experience': self._determine_experience_level(job),
                    'skills': self._extract_skills(job.get('description', '')),
                    'matchScore': self._calculate_match_score(job),
                    'postedDate': self._format_posted_date(job)
                }
                
                # Filter jobs based on ALL criteria (AND logic, not OR)
                should_include = True
                
                # Filter by industry (if multiple industries selected)
                if params and 'industry' in params and params['industry'] and len(params['industry']) > 0:
                    if transformed_job['industry'] not in params['industry']:
                        should_include = False
                
                # Filter by location (if multiple locations selected)
                if params and 'location' in params and params['location'] and len(params['location']) > 1:
                    if transformed_job['location'] not in params['location']:
                        should_include = False
                
                # Filter by experience
                if params and 'experience' in params and params['experience'] and len(params['experience']) > 0:
                    if transformed_job['experience'] not in params['experience']:
                        should_include = False
                
                # Filter by job type
                if params and 'jobType' in params and params['jobType'] and len(params['jobType']) > 0:
                    if transformed_job['jobType'] not in params['jobType']:
                        should_include = False
                
                if should_include:
                    transformed_jobs.append(transformed_job)
            
            return transformed_jobs
        
        except requests.RequestException as e:
            print(f"Error fetching jobs: {e}")
            if hasattr(e, 'response'):
                print("Response content:", e.response.text)
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def _determine_industry(self, job: Dict, adzuna_category: str) -> str:
        """Determine industry based on Adzuna category with cosine similarity fallback"""
        # First, try to map directly from Adzuna category
        if adzuna_category in self.category_to_industry:
            return self.category_to_industry[adzuna_category]
        
        # Prepare text for analysis - combine title and description in Spanish
        title = job.get('title', '')
        description = job.get('description', '')
        combined_text = f"{title} {description}"
        
        # Use cosine similarity to determine industry
        industry, similarity = self.industry_classifier.predict_industry(combined_text)
        
        # If similarity is high enough, use the result
        if similarity > 0.15:  # Similarity threshold can be adjusted
            return industry
        
        # If similarity is too low, fall back to keyword-based approach (in Spanish)
        category_label = job.get('category', {}).get('label', '').lower()
        job_title = job.get('title', '').lower()
        
        # Try to identify the industry from Spanish keywords in category label or title
        if any(tech in category_label or tech in job_title for tech in ['it', 'software', 'tech', 'tecnología', 'informática', 'programador', 'desarrollador', 'sistemas']):
            return 'Tecnología'
        elif any(finance in category_label or finance in job_title for finance in ['finanzas', 'financiero', 'contable', 'contabilidad', 'fiscal', 'banca']):
            return 'Finanzas'
        elif any(consult in category_label or consult in job_title for consult in ['consultoría', 'consultor', 'asesoría', 'asesor']):
            return 'Consultoría'
        elif any(marketing in category_label or marketing in job_title for marketing in ['marketing', 'publicidad', 'digital', 'seo', 'sem', 'redes sociales']):
            return 'Marketing'
        elif any(sales in category_label or sales in job_title for sales in ['ventas', 'comercial', 'sales', 'vendedor', 'comercio']):
            return 'Ventas'
        elif any(health in category_label or health in job_title for health in ['salud', 'médico', 'sanitario', 'enfermero', 'farmacia', 'clínica', 'hospital']):
            return 'Salud'
        elif any(edu in category_label or edu in job_title for edu in ['educación', 'profesor', 'docente', 'maestro', 'formación', 'enseñanza']):
            return 'Educación'
        elif any(eng in category_label or eng in job_title for eng in ['ingeniería', 'ingeniero', 'civil', 'industrial', 'mecánico', 'eléctrico']):
            return 'Ingeniería'
        elif any(design in category_label or design in job_title for design in ['diseño', 'diseñador', 'gráfico', 'creativo', 'ui', 'ux']):
            return 'Diseño'
        elif any(hr in category_label or hr in job_title for hr in ['recursos humanos', 'rrhh', 'talento', 'selección', 'personal']):
            return 'Recursos Humanos'
        
        # Default to the cosine similarity result if everything else fails
        return industry

    def _format_location(self, job: Dict) -> str:
        """Format location from job data"""
        location_areas = job.get('location', {}).get('area', [])
        # Get city name (usually the most specific location)
        if location_areas and len(location_areas) > 0:
            return location_areas[-1]
        return 'Spain'

    def _format_salary(self, job: Dict) -> str:
        """Format salary information"""
        min_salary = job.get('salary_min', 0)
        max_salary = job.get('salary_max', 0)
        
        if min_salary and max_salary:
            return f"{int(min_salary):,}€ - {int(max_salary):,}€"
        return None

    def _format_posted_date(self, job: Dict) -> str:
        """Format posted date"""
        # Use the created timestamp if available
        created = job.get('created', '')
        if created:
            # For now using a simplified approach
            return f"Hace {random.randint(1, 30)} días"
        return "Hace poco"

    def _calculate_match_score(self, job: Dict) -> int:
        """
        Calculate a match score based on job attributes
        This is a placeholder - you'll want to replace with more sophisticated AI matching
        """
        relevance = random.randint(20, 90)  # Simulated for now
        return min(relevance, 100)

    def _extract_skills(self, description: str) -> List[str]:
        """Extract key skills from job description"""
        skill_keywords = {
            'tech': ['python', 'javascript', 'react', 'sql', 'agile', 'typescript', 'java', 'node', 'aws', 'cloud'],
            'finance': ['excel', 'contabilidad', 'fiscal', 'finanzas', 'sap', 'erp'],
            'consulting': ['consultoría', 'proyectos', 'gestión', 'negocio', 'procesos'],
            'marketing': ['seo', 'sem', 'redes sociales', 'marketing digital', 'analytics', 'contenidos'],
            'sales': ['ventas', 'comercial', 'negociación', 'cliente', 'crm', 'atención al cliente'],
            'hr': ['selección', 'reclutamiento', 'talento', 'formación', 'recursos humanos', 'nóminas']
        }
        
        found_skills = []
        description_lower = description.lower()
        for category, skills in skill_keywords.items():
            found_skills.extend([skill for skill in skills if skill.lower() in description_lower])
        
        return list(set(found_skills))[:3]

    def _determine_job_type(self, job: Dict) -> str:
        """Determine job type based on job data"""
        contract_type = job.get('contract_type', '').lower()
        if 'full' in contract_type or 'tiempo completo' in contract_type or 'jornada completa' in contract_type:
            return 'Tiempo completo'
        elif 'part' in contract_type or 'medio tiempo' in contract_type or 'parcial' in contract_type or 'media jornada' in contract_type:
            return 'Medio tiempo'
        elif 'contract' in contract_type or 'contrato' in contract_type or 'temporal' in contract_type:
            return 'Contrato'
        elif 'freelance' in contract_type or 'autónomo' in contract_type:
            return 'Freelance'
        elif 'internship' in contract_type or 'prácticas' in contract_type or 'becario' in contract_type:
            return 'Prácticas'
        
        # Try to determine from title or description if contract_type is not available
        description = job.get('description', '').lower()
        title = job.get('title', '').lower()
        
        if any(full in description or full in title for full in ['tiempo completo', 'jornada completa', 'full time']):
            return 'Tiempo completo'
        elif any(part in description or part in title for part in ['tiempo parcial', 'media jornada', 'part time']):
            return 'Medio tiempo'
        elif any(contract in description or contract in title for contract in ['temporal', 'por proyecto']):
            return 'Contrato'
        elif any(free in description or free in title for free in ['freelance', 'autónomo', 'por cuenta propia']):
            return 'Freelance'
        elif any(intern in description or intern in title for intern in ['prácticas', 'becario', 'internship']):
            return 'Prácticas'
        
        return 'Tiempo completo'  # Default

    def _determine_experience_level(self, job: Dict) -> str:
        """Determine experience level based on job data"""
        description = job.get('description', '').lower()
        title = job.get('title', '').lower()
        
        # Check for experience keywords in title and description (Spanish-focused)
        if any(senior in title or senior in description for senior in ['senior', 'sr', 'líder', 'lead', 'jefe', 'responsable', 'manager']):
            return 'Senior'
        elif any(mid in title or mid in description for mid in ['medio', 'intermedio', 'experienced', 'con experiencia']):
            return 'Intermedio'
        elif any(junior in title or junior in description for junior in ['junior', 'jr', 'entry', 'recién', 'trainee']):
            return 'Junior'
        elif any(exec in title or exec in description for exec in ['ejecutivo', 'director', 'gerente', 'directivo']):
            return 'Directivo'
        elif any(beginner in title or beginner in description for beginner in ['principiante', 'sin experiencia', 'trainee', 'graduado']):
            return 'Principiante'
        
        # Try to determine from years of experience if mentioned (Spanish patterns)
        experience_patterns = [
            r'(\d+)\+?\s*años?\s+de\s+experiencia',
            r'(\d+)-(\d+)\s*años?\s+de\s+experiencia',
            r'experiencia\s+de\s+(\d+)\+?\s*años?',
            r'experiencia\s+mínima\s+de\s+(\d+)\s*años?',
            r'al\s+menos\s+(\d+)\s*años?\s+de\s+experiencia',
            r'(\d+)\+?\s*years?\s+of\s+experience',
            r'(\d+)-(\d+)\s*years?\s+of\s+experience',
            r'experience\s+of\s+(\d+)\+?\s*years?'
        ]
        
        for pattern in experience_patterns:
            matches = re.search(pattern, description)
            if matches:
                # Extract the years mentioned
                years = int(matches.group(1))
                if years < 1:
                    return 'Principiante'
                elif years < 3:
                    return 'Junior'
                elif years < 5:
                    return 'Intermedio'
                else:
                    return 'Senior'
        
        return 'Intermedio'  # Default  # Default