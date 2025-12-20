"use client"

import { LandingNav } from "@/components/landing/landing-nav"
import { Footer } from "@/components/landing/footer"

export default function PrivacyPage() {
  return (
    <main className="min-h-screen bg-black">
      <LandingNav />
      <div className="pt-16 pb-24">
        <div className="max-w-4xl mx-auto px-4 py-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 text-center">
            Privacy Policy
          </h1>
          <p className="text-gray-400 text-sm text-center mb-12">Last updated: December 2024</p>
          
          <div className="space-y-8">
            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">1. Introduction</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  Courtvision ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy
                  explains how we collect, use, disclose, and safeguard your information when you use our tennis
                  analytics platform and services (the "Service").
                </p>
                <p>
                  By using our Service, you agree to the collection and use of information in accordance with this
                  policy. If you do not agree with our policies and practices, do not use our Service.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">2. Information We Collect</h2>
              
              <h3 className="text-xl font-medium text-white mb-3 mt-6">2.1 Information You Provide</h3>
              <div className="space-y-3 text-gray-400">
                <p>We collect information that you provide directly to us, including:</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Account Information:</strong> Name, email address, password, and role (coach or player)</li>
                  <li><strong>Profile Information:</strong> Any additional profile details you choose to provide</li>
                  <li><strong>Match Data:</strong> Playsight video links, player identifications, and match-related information</li>
                  <li><strong>Team Information:</strong> Team names, codes, and membership details</li>
                  <li><strong>Communication Data:</strong> Messages, support requests, and other communications with us</li>
                  <li><strong>Payment Information:</strong> Billing details and payment method information (processed through third-party payment processors)</li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-white mb-3 mt-6">2.2 Automatically Collected Information</h3>
              <div className="space-y-3 text-gray-400">
                <p>When you use our Service, we automatically collect certain information, including:</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Usage Data:</strong> Pages visited, features used, time spent, and interaction patterns</li>
                  <li><strong>Device Information:</strong> Device type, operating system, browser type, and device identifiers</li>
                  <li><strong>Log Data:</strong> IP address, access times, error logs, and system activity</li>
                  <li><strong>Location Data:</strong> General location information derived from IP address (not precise GPS data)</li>
                  <li><strong>Cookies and Tracking:</strong> Information collected through cookies, web beacons, and similar technologies</li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-white mb-3 mt-6">2.3 Information from Third Parties</h3>
              <div className="space-y-3 text-gray-400">
                <p>We may receive information about you from third-party services:</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Authentication Providers:</strong> Information from Supabase Auth when you sign up or log in</li>
                  <li><strong>Video Platforms:</strong> Metadata from Playsight videos you link to our Service</li>
                  <li><strong>Analytics Services:</strong> Aggregated usage statistics and performance metrics</li>
                </ul>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">3. How We Use Your Information</h2>
              <div className="space-y-3 text-gray-400">
                <p>We use the information we collect for the following purposes:</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Service Provision:</strong> To provide, maintain, operate, and improve our Service</li>
                  <li><strong>Video Processing:</strong> To process and analyze your match videos using computer vision technology</li>
                  <li><strong>Account Management:</strong> To create and manage your account, authenticate users, and process transactions</li>
                  <li><strong>Team Features:</strong> To facilitate team creation, member management, and team-based analytics</li>
                  <li><strong>Communication:</strong> To send you technical notices, updates, security alerts, and support messages</li>
                  <li><strong>Customer Support:</strong> To respond to your inquiries, comments, and requests</li>
                  <li><strong>Analytics:</strong> To understand how users interact with our Service and improve user experience</li>
                  <li><strong>Legal Compliance:</strong> To comply with legal obligations, enforce our terms, and protect our rights</li>
                  <li><strong>Security:</strong> To detect, prevent, and address fraud, security issues, and technical problems</li>
                  <li><strong>Marketing:</strong> To send you promotional communications (with your consent, and you may opt out at any time)</li>
                </ul>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">4. Data Sharing and Disclosure</h2>
              
              <h3 className="text-xl font-medium text-white mb-3 mt-6">4.1 We Do Not Sell Your Data</h3>
              <p className="text-gray-400 mb-4">
                We do not sell, rent, or trade your personal information to third parties for their marketing purposes.
              </p>

              <h3 className="text-xl font-medium text-white mb-3 mt-6">4.2 Sharing with Your Consent</h3>
              <p className="text-gray-400 mb-4">
                We may share your information when you explicitly consent to such sharing, such as when you choose to
                share match data with team members or make your profile public.
              </p>

              <h3 className="text-xl font-medium text-white mb-3 mt-6">4.3 Service Providers</h3>
              <div className="space-y-3 text-gray-400">
                <p>We may share your information with trusted third-party service providers who assist us in operating our Service:</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Hosting and Infrastructure:</strong> Supabase (database, authentication, hosting)</li>
                  <li><strong>Payment Processing:</strong> Third-party payment processors (Stripe, PayPal, etc.)</li>
                  <li><strong>Analytics:</strong> Service analytics providers to help us understand usage patterns</li>
                  <li><strong>Email Services:</strong> Email delivery services for transactional and marketing emails</li>
                  <li><strong>Video Processing:</strong> Cloud computing services for video analysis (if applicable)</li>
                </ul>
                <p className="mt-3">
                  These service providers are contractually obligated to protect your information and use it only for
                  the purposes we specify. They are prohibited from using your information for their own purposes.
                </p>
              </div>

              <h3 className="text-xl font-medium text-white mb-3 mt-6">4.4 Legal Requirements</h3>
              <div className="space-y-3 text-gray-400">
                <p>We may disclose your information if required by law or in response to valid requests by public authorities:</p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>To comply with legal obligations, court orders, or legal processes</li>
                  <li>To respond to government requests or regulatory inquiries</li>
                  <li>To enforce our Terms of Service or other agreements</li>
                  <li>To protect our rights, property, or safety, or that of our users or others</li>
                  <li>To investigate fraud, security breaches, or other illegal activities</li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-white mb-3 mt-6">4.5 Business Transfers</h3>
              <p className="text-gray-400">
                In the event of a merger, acquisition, reorganization, bankruptcy, or sale of assets, your information
                may be transferred as part of that transaction. We will notify you of any such change in ownership
                or control of your personal information.
              </p>

              <h3 className="text-xl font-medium text-white mb-3 mt-6">4.6 Aggregated and Anonymized Data</h3>
              <p className="text-gray-400">
                We may share aggregated, anonymized, or de-identified information that cannot reasonably be used to
                identify you. This data may be used for research, analytics, or other purposes.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">5. Data Security</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  We implement comprehensive technical and organizational security measures to protect your personal
                  information against unauthorized access, alteration, disclosure, or destruction:
                </p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Encryption:</strong> All data is encrypted in transit using TLS/SSL and at rest using industry-standard encryption</li>
                  <li><strong>Access Controls:</strong> Strict access controls and authentication requirements for our systems</li>
                  <li><strong>Database Security:</strong> Row-level security policies and secure database configurations</li>
                  <li><strong>Regular Audits:</strong> Security audits and vulnerability assessments</li>
                  <li><strong>Secure Infrastructure:</strong> Hosting on secure, compliant cloud infrastructure (Supabase)</li>
                  <li><strong>Employee Training:</strong> Security training for employees with access to personal data</li>
                  <li><strong>Incident Response:</strong> Procedures for detecting, reporting, and responding to security incidents</li>
                </ul>
                <p className="mt-3">
                  However, no method of transmission over the Internet or electronic storage is 100% secure. While we
                  strive to use commercially acceptable means to protect your information, we cannot guarantee absolute
                  security. You are responsible for maintaining the confidentiality of your account credentials.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">6. Your Rights and Choices</h2>
              <div className="space-y-3 text-gray-400">
                <p>Depending on your location, you may have the following rights regarding your personal information:</p>
                
                <h3 className="text-xl font-medium text-white mb-3 mt-6">6.1 Access and Portability</h3>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Request access to your personal data</li>
                  <li>Request a copy of your data in a portable, machine-readable format</li>
                  <li>View and download your match data and statistics</li>
                </ul>

                <h3 className="text-xl font-medium text-white mb-3 mt-6">6.2 Correction and Updates</h3>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Update or correct inaccurate personal information</li>
                  <li>Modify your account settings and preferences</li>
                </ul>

                <h3 className="text-xl font-medium text-white mb-3 mt-6">6.3 Deletion</h3>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Request deletion of your personal data</li>
                  <li>Delete your account and all associated data</li>
                  <li>Note: Some information may be retained as required by law or for legitimate business purposes</li>
                </ul>

                <h3 className="text-xl font-medium text-white mb-3 mt-6">6.4 Objection and Restriction</h3>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Object to processing of your personal data for certain purposes</li>
                  <li>Request restriction of processing in certain circumstances</li>
                </ul>

                <h3 className="text-xl font-medium text-white mb-3 mt-6">6.5 Marketing Communications</h3>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Opt out of marketing emails by clicking the unsubscribe link in any marketing email</li>
                  <li>Manage your communication preferences in your account settings</li>
                  <li>Note: You may still receive transactional emails related to your account</li>
                </ul>

                <h3 className="text-xl font-medium text-white mb-3 mt-6">6.6 Exercising Your Rights</h3>
                <p>
                  To exercise any of these rights, please contact us through the contact form in the footer or
                  email us directly. We will respond to your request within 30 days, subject to verification of your
                  identity and any applicable legal requirements.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">7. Data Retention</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  We retain your personal information for as long as necessary to fulfill the purposes outlined in this
                  Privacy Policy, unless a longer retention period is required or permitted by law:
                </p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Account Data:</strong> Retained while your account is active and for a reasonable period after account deletion for legal and business purposes</li>
                  <li><strong>Match Data:</strong> Retained until you delete the match or your account is deleted</li>
                  <li><strong>Transaction Records:</strong> Retained as required by law (typically 7 years for tax and accounting purposes)</li>
                  <li><strong>Log Data:</strong> Retained for security and troubleshooting purposes (typically 90 days to 1 year)</li>
                  <li><strong>Marketing Data:</strong> Retained until you opt out or request deletion</li>
                </ul>
                <p className="mt-3">
                  When we delete your data, we use secure deletion methods. Some data may remain in backup systems
                  for a limited time before being permanently deleted.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">8. Cookies and Tracking Technologies</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  We use cookies, web beacons, and similar tracking technologies to collect and store information
                  about your use of our Service:
                </p>
                
                <h3 className="text-xl font-medium text-white mb-3 mt-6">8.1 Types of Cookies</h3>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Essential Cookies:</strong> Required for the Service to function (authentication, security)</li>
                  <li><strong>Functional Cookies:</strong> Remember your preferences and settings</li>
                  <li><strong>Analytics Cookies:</strong> Help us understand how users interact with our Service</li>
                  <li><strong>Marketing Cookies:</strong> Used to deliver relevant advertisements (with your consent)</li>
                </ul>

                <h3 className="text-xl font-medium text-white mb-3 mt-6">8.2 Cookie Management</h3>
                <p>
                  You can control cookies through your browser settings. Most browsers allow you to refuse or delete
                  cookies. However, disabling certain cookies may limit your ability to use some features of our Service.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">9. Children's Privacy</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  Our Service is not intended for children under the age of 13 (or the minimum age in your jurisdiction).
                  We do not knowingly collect personal information from children under 13.
                </p>
                <p>
                  If you are a parent or guardian and believe your child has provided us with personal information,
                  please contact us immediately. If we become aware that we have collected personal information from
                  a child under 13 without parental consent, we will take steps to delete that information promptly.
                </p>
                <p>
                  For users between 13 and 18, we recommend parental supervision and consent when using our Service.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">10. International Data Transfers</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  Your information may be transferred to and processed in countries other than your country of residence.
                  These countries may have data protection laws that differ from those in your country.
                </p>
                <p>
                  By using our Service, you consent to the transfer of your information to countries outside your
                  jurisdiction, including the United States, where our service providers (such as Supabase) may be located.
                </p>
                <p>
                  We ensure that appropriate safeguards are in place for such transfers, including:
                </p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Standard contractual clauses approved by data protection authorities</li>
                  <li>Certification under recognized data protection frameworks (e.g., EU-US Privacy Shield successor frameworks)</li>
                  <li>Compliance with applicable data protection laws</li>
                </ul>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">11. California Privacy Rights (CCPA)</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  If you are a California resident, you have additional rights under the California Consumer Privacy Act (CCPA):
                </p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Right to Know:</strong> Request disclosure of categories and specific pieces of personal information collected</li>
                  <li><strong>Right to Delete:</strong> Request deletion of personal information (subject to certain exceptions)</li>
                  <li><strong>Right to Opt-Out:</strong> Opt out of the sale of personal information (we do not sell personal information)</li>
                  <li><strong>Non-Discrimination:</strong> We will not discriminate against you for exercising your privacy rights</li>
                </ul>
                <p className="mt-3">
                  To exercise your California privacy rights, contact us through the contact form or email us directly.
                  We will verify your identity before processing your request.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">12. European Privacy Rights (GDPR)</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  If you are located in the European Economic Area (EEA), United Kingdom, or Switzerland, you have
                  additional rights under the General Data Protection Regulation (GDPR):
                </p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Right of Access:</strong> Obtain confirmation and access to your personal data</li>
                  <li><strong>Right to Rectification:</strong> Correct inaccurate or incomplete data</li>
                  <li><strong>Right to Erasure:</strong> Request deletion of your personal data ("right to be forgotten")</li>
                  <li><strong>Right to Restrict Processing:</strong> Limit how we process your data</li>
                  <li><strong>Right to Data Portability:</strong> Receive your data in a structured, machine-readable format</li>
                  <li><strong>Right to Object:</strong> Object to processing based on legitimate interests</li>
                  <li><strong>Right to Withdraw Consent:</strong> Withdraw consent for processing where consent is the legal basis</li>
                  <li><strong>Right to Lodge a Complaint:</strong> File a complaint with your local data protection authority</li>
                </ul>
                <p className="mt-3">
                  Our legal basis for processing your data includes: (1) your consent, (2) performance of a contract,
                  (3) compliance with legal obligations, (4) protection of vital interests, (5) performance of a
                  task in the public interest, and (6) legitimate interests.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">13. Third-Party Links and Services</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  Our Service may contain links to third-party websites, services, or applications that are not owned
                  or controlled by us. We are not responsible for the privacy practices of these third parties.
                </p>
                <p>
                  When you click on a third-party link, you will be directed to that third party's site. We strongly
                  advise you to review the Privacy Policy of every site you visit.
                </p>
                <p>
                  We have no control over and assume no responsibility for the content, privacy policies, or practices
                  of any third-party sites or services.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">14. Data Breach Notification</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  In the event of a data breach that may compromise your personal information, we will:
                </p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Investigate the breach and take immediate steps to contain and remediate it</li>
                  <li>Notify affected users within 72 hours of becoming aware of the breach (as required by applicable law)</li>
                  <li>Notify relevant data protection authorities as required by law</li>
                  <li>Provide clear information about what data was affected and what steps we are taking</li>
                  <li>Offer guidance on steps you can take to protect yourself</li>
                </ul>
                <p className="mt-3">
                  Notification will be sent to the email address associated with your account or through other
                  reasonable means.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">15. Changes to This Privacy Policy</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  We may update our Privacy Policy from time to time. We will notify you of any material changes by:
                </p>
                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li>Posting the new Privacy Policy on this page</li>
                  <li>Updating the "Last updated" date at the top of this page</li>
                  <li>Sending you an email notification (for material changes)</li>
                  <li>Displaying a prominent notice on our Service (for significant changes)</li>
                </ul>
                <p className="mt-3">
                  Material changes will become effective 30 days after notification, unless otherwise stated. Your
                  continued use of our Service after the effective date of any changes constitutes acceptance of the
                  updated Privacy Policy.
                </p>
                <p>
                  If you do not agree with the changes, you may close your account and stop using our Service.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">16. Data Controller and Contact Information</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  For the purposes of applicable data protection laws, Courtvision is the data controller responsible
                  for your personal information.
                </p>
                <p>
                  If you have any questions, concerns, or requests regarding this Privacy Policy or our data practices,
                  please contact us:
                </p>
                <ul className="list-none space-y-2 ml-4 mt-4">
                  <li className="text-white font-medium">Courtvision</li>
                  <li>Email: Use the contact form in the footer of our website</li>
                  <li>Website: courtvision.com</li>
                </ul>
                <p className="mt-4">
                  For EU residents, you also have the right to lodge a complaint with your local data protection
                  authority if you believe we have not addressed your concerns adequately.
                </p>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">17. Consent and Agreement</h2>
              <div className="space-y-3 text-gray-400">
                <p>
                  By using our Service, you consent to the collection, use, and disclosure of your information as
                  described in this Privacy Policy.
                </p>
                <p>
                  If you do not agree with any part of this Privacy Policy, you must not use our Service. If you
                  have already created an account and no longer agree with this Privacy Policy, you may delete your
                  account at any time through your account settings.
                </p>
                <p>
                  This Privacy Policy is incorporated into and forms part of our Terms of Service. By agreeing to
                  our Terms of Service, you also agree to this Privacy Policy.
                </p>
              </div>
            </section>
          </div>
        </div>
      </div>
      <Footer />
    </main>
  )
}
