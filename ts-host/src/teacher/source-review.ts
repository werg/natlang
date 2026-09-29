/** Source annotation disputes stay out of generation and training until adjudicated. */
export type SourceReview = {
  dataset: string;
  id: string;
  aliases: readonly string[];
  text: string;
  annotatedLabel: string;
  status: 'pending' | 'resolved';
  reason: string;
};

/** IDs use model-visible text; aliases preserve the earlier text+label identity. */
export const SOURCE_REVIEWS: readonly SourceReview[] = [
  {"dataset": "scifact", "id": "claim:79", "aliases": [], "text": "Active caspase-11 protein promotes pyroptosis.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract5099266 studies phagosome/lysosome fusion and caspase activation, but never establishes pyroptosis. The required cell-death mechanism would need outside evidence. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:104", "aliases": [], "text": "Allogeneic mechanical circulatory support is not as effective as autologous mechanical circulatory support for treating acute myocardial infarction.", "annotatedLabel": "CONTRADICT", "status": "pending", "reason": "Abstract40164383 compares mesenchymal stem cell injections for ischemic cardiomyopathy. The claim expands MSC to mechanical circulatory support and changes the clinical condition to acute myocardial infarction. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:160", "aliases": [], "text": "Bacterial meningitis can be diagnosed on the basis of positive cerebrospinal fluid (CSF) cultures.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract52874170 establishes biochemical CSF markers but does not discuss positive CSF cultures. A medically plausible assertion is not sufficient document-only evidence. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:164", "aliases": [], "text": "Bariatric surgery increases rates of colorectal cancer.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract5824985 reports no detected association with cancer overall and gives no colorectal cancer outcome. It cannot justify the annotated increase in colorectal cancer. This is an evidence gap, not proof of the opposite subtype effect. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:165", "aliases": [], "text": "Bariatric surgery increases rates of postmenopausal breast cancer.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract5824985 reports no detected association with cancer overall and gives no postmenopausal breast cancer outcome. It cannot justify the annotated increase. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:168", "aliases": [], "text": "Bariatric surgery reduces colorectal cancer.", "annotatedLabel": "CONTRADICT", "status": "pending", "reason": "Abstract5824985 reports no detected association with cancer overall, not a colorectal-specific result. Absence of an aggregate association does not establish contradiction of a subtype reduction. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:169", "aliases": [], "text": "Bariatric surgery reduces postmenopausal breast cancer.", "annotatedLabel": "CONTRADICT", "status": "pending", "reason": "Abstract5824985 reports no detected association with cancer overall, not a postmenopausal breast cancer result. Absence of an aggregate association does not establish contradiction of a subtype reduction. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:200", "aliases": [], "text": "CD28 initiates tonic signaling in conventional T cells, which causes an exhaustion phenotype and limited efficiency.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract18231807 attributes tonic signaling and exhaustion to engineered CAR T cells, and CD28 augments it. The claim instead attributes initiation to CD28 in conventional T cells. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:258", "aliases": [], "text": "Cholesterol loading induces KLF4 expression in VSMCs, resulting in the expression of pro-inflammatory cytokines.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract22080671 establishes KLF4-dependent phenotype transitions and KLF4 targets in cholesterol-treated SMCs, but not the asserted induction of KLF4 expression by cholesterol followed by cytokine expression. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:331", "aliases": [], "text": "Deltex interacts with eIF3. There is no known interaction between Deltex and elF3", "annotatedLabel": "CONTRADICT", "status": "pending", "reason": "The source claim simultaneously asserts and denies a Deltex/eIF3 interaction. Abstract9505448 supports bridging recruitment, but this internally contradictory compound claim is unsuitable for the binary oracle. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:372", "aliases": [], "text": "Eilat virus (EILV) produced in mosquitos elicits rapid and long-lasting neutralizing antibodies in nonhuman primates.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract24922825 reports long-lasting neutralizing antibodies in mice, while the nonhuman-primate result reports rapid robust protective immunity without antibody duration. The claim transfers the mouse duration to primates and omits the CHIKV chimera. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:397", "aliases": [], "text": "Exercise reduces cancer mortality rates among Chinese citizens.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract1456068 reports combined lifestyle-score associations with cancer mortality, not an isolated exercise effect. Five independently mortality-associated factors do not establish an exercise-specific cancer mortality effect. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:564", "aliases": [], "text": "In British Men, haplogroup I increases risk of cardiovascular disease by 50%.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract2867345 quantifies about 50% higher coronary artery disease risk. The claim assigns this magnitude to cardiovascular disease generally; the broader quantified endpoint is not established. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:584", "aliases": [], "text": "In rhesus macaques, daily subcutaneous injections of tenofovir protects against rectally transmitted simian-human immunodeficiency virus.", "annotatedLabel": "SUPPORT", "status": "pending", "reason": "Abstract14260013 tests subcutaneous tenofovir together with emtricitabine, while the single-agent subcutaneous arm uses emtricitabine. It does not isolate protection by daily subcutaneous tenofovir. Preserve source gold pending independent adjudication."},
  {"dataset": "scifact", "id": "claim:747", "aliases": [], "text": "MafA ubiquitination decreases the recruitment of coavtivator P/CAF by MafA.", "annotatedLabel": "CONTRADICT", "status": "pending", "reason": "Abstract11291348 attributes recruitment to phosphorylation and says release of P/CAF precedes ubiquitination. It does not establish whether ubiquitination itself decreases recruitment or directly contradict that causal claim. Preserve source gold pending independent adjudication."},
  { dataset: 'scifact', id: 'claim:657', aliases: [],
    text: 'Intramembrane cleavage by signal peptide peptidase aids in the degradation of proteins with a complex membrane orientation.',
    annotatedLabel: 'SUPPORT', status: 'pending',
    reason: 'Supplied abstract8533245 establishes RHBDL4 rhomboid protease cleavage, not signal peptide peptidase. Replacing the named mechanism requires outside evidence. Hold the original label rather than penalizing the distinction.' },
  { dataset: 'scifact', id: 'claim:748', aliases: [],
    text: 'MafA ubiquitination increases the recruitment of coavtivator P/CAF by MafA.',
    annotatedLabel: 'SUPPORT', status: 'pending',
    reason: 'Supplied abstract11291348 attributes P/CAF recruitment to GSK-3-mediated phosphorylation, and says P/CAF protects from ubiquitination. It does not establish the claimed ubiquitination cause. Preserve source gold pending adjudication.' },
  { dataset: 'folio', id: 'story:406', aliases: ['folio:story:406', 'story406'],
    text: 'People either regularly drink coffee or joke about being addicted to caffeine.',
    annotatedLabel: 'C1=True; C2=True; C3=False; C4=True', status: 'pending',
    reason: 'A person Rina who drinks coffee, depends on caffeine, is a student and is unaware caffeine is a drug, but does not joke about addiction, satisfies all five English premises and makes C3 true. The final conditional has a false antecedent under either not-both or non-dependent-student reading. The asserted contradiction therefore needs source adjudication.' },
  { dataset: 'scifact', id: 'claim:466', aliases: [],
    text: "Genomic sequences involved in alternative splicing responsible for Hutchinson-Gilford progeria syndrome (HGPS) are abundant in the ''progerinonly'' allele of Lmna knock-in models.",
    annotatedLabel: 'CONTRADICT', status: 'pending',
    reason: 'The supplied abstract22544171 describes splicing correction in human fibroblasts, not progerinonly alleles or Lmna knock-in models. Its annotated sentences do not establish the claim or its negation. Hold the original label until sufficient source evidence is independently established.' },
  { dataset: 'scifact', id: 'claim:576', aliases: [],
    text: 'In melanoma, anti-CTLA-4 treatment reinvigorates exhausted PD-1+Eomes+CD8 T cells.',
    annotatedLabel: 'SUPPORT', status: 'pending',
    reason: 'The supplied abstract4468861 attributes reversal of exhaustion to addition of PD-L1 blockade, while anti-CTLA4 predominantly inhibits Treg cells. The annotated sentence about resistance due to PD-L1 does not establish the claimed anti-CTLA4 effect or the specified phenotype. Hold without relabeling.' },
  { dataset: 'folio', id: 'story:24', aliases: ['folio:story:24', 'story24'],
    text: 'All Leetcode problems that are recommended to novices are easy. 2Sum is recommended to novices.',
    annotatedLabel: '2Sum easy Leetcode=True; 4Sum novice Leetcode=False; 2Sum AC above20=False', status: 'pending',
    reason: 'English rules require Leetcode-problem membership, but no premise establishes that membership for 2Sum or4Sum. Non-Leetcode named objects with the stated recommendation/star predicates satisfy the guarded rules while falsifying the asserted entailment. Source labels also require exclusive easy/hard and an AC-rate interpretation. Preserve labels and hold the whole story.' },
  {
    dataset: 'tatqa', id: 'f944b361-6e00-45c8-a7e1-1f5c6e0fd6b1', aliases: [],
    text: 'What is the average Selling, general and administrative expenses for the period December 31, 2019 and September 29, 2019?',
    annotatedLabel: '276 million', status: 'pending',
    reason: 'The supplied accounting table shows (285) and (267). The question does not specify whether to average signed values (-276) or expense magnitudes (276). Hold until the convention is independently established; do not penalize either interpretation or rewrite source gold.',
  },
  {
    dataset: 'banking77',
    id: 'a1658811502225e680d921d1ba4aa4492cc1b7225160f46c1cc360393d0f2fcd',
    aliases: ['a9a729504209367ae5cdc1fba30d835718922358d51a8f922b205513a6925935'],
    text: 'I did not get the cash that is showing up in my app.',
    annotatedLabel: 'cash_withdrawal_not_recognised',
    status: 'pending',
    reason: 'Cash not received does not establish whether the withdrawal was unrecognized or failed.',
  },
  {
    dataset: 'banking77',
    id: '36796a0f7fcdef9e1f89b825bb4150ac5d530bc865d3da2a6f5b7902298c2acb',
    aliases: ['20d4392399dd4a19d86356619825f146f3b7b3bbbfeaa626c493f50ae5abc002'],
    text: 'My app says I withdraw funds from my account through an ATM.',
    annotatedLabel: 'cash_withdrawal_not_recognised',
    status: 'pending',
    reason: 'The visible statement does not establish whether the recorded withdrawal is disputed.',
  },
  {
    dataset: 'banking77',
    id: 'e8201faf7e42751836cf4ae49618b262fa23d3c7cc3783a7ec877acda07c69fe',
    aliases: ['8036fcc7415fdfebff9c8c50b5afeb7886d3f39f67a328a46b4ef34caa9b1fd5'],
    text: 'How can I dispute a debit transaction?',
    annotatedLabel: 'direct_debit_payment_not_recognised',
    status: 'pending',
    reason: 'A disputed debit transaction does not establish direct debit or lack of authorization.',
  },
  ...[
    ['db49742d7d0f0a9986b4fe7c04a26f815228b6ce811fc45d0c4ef7eb9410da7a',
      "Ellie's car broke down near a tunnel.\nBy the time she left the area she counted 274 cars.\nEveryone had to evacuate the tunnel on foot.\nEllie decided to take a nap in her car.", 'a',
      'Neither hypothesis explains the stated counting; the missing connecting events leave the preference unsupported.'],
    ['921971912e8a89bbed73f18ab99389871fa1a3be590f5a5e1b592d29e76d78ac',
      'Rodney was looking forward to a date with his girlfriend all day.\nIn the end Rodney got to go on a wonderful date with his girlfriend.\nHe ended up having to work later than expected.\nHe ended up getting off of work earlier than expected.', 'a',
      'The visible endpoints do not establish why later work is preferred to getting off early.'],
    ['172063d25cc1bc24e7de5316084a2438b487156042279e3a4a20ef3ab969de4b',
      'Sandy was really disappointed no one remembered her birthday.\nShe arrived to find all of her friends there for her surprise party.\nSandy came home from a long day sad and ready to exercise.\nSandy came home from a long day sad and ready to go to bed.', 'b',
      'Both alternatives bring Sandy home; the endpoints do not distinguish exercising from sleeping.'],
    ['e472f170b5649e1c02d5b1044a258761ab3dabac57b0811dae90a2e02eef8671',
      'Jesse leans against the plain wooden foot board of her bed.\nShe finishes with a lonely sigh, pining for her recently dead mom.\nJesse folded the worn quilts lying there.\nJesse folded the worn quilts for her mother.', 'a',
      'Both quilting actions can precede remembering her mother; the intended preference needs source adjudication.'],
    ['e2c9a606854c4c0007362b26c90647a0921098ad4b8c0ff00a7874ff641f395b',
      "Emily had an essay due the next day.\nEmily's mother forced her to finish chores.\nEmily stopped washing dishes and started writing.\nEmily started washing dishes an stopped writing.", 'a',
      'Both alternatives can lead to the stated chores; the annotated preference depends on unstated timing.'],
    ['33635093f35a02a840c54dbd73818a740e8398ea276be48a1875e9643ceb4b16',
      'Todd hated making coffee in the morning.\nTodd happily enjoyed his life saving gift.\nTodd forgot to make coffee and was tired driving in to work.\nTodd made coffee anyway to that he would be alert driving in to work.', 'b',
      'The gift and its connection to either hypothesis are missing from the visible story.'],
    ['3b1ce08a76ad9479c5855708987d0ae57667d728dc72d599a32d76e44bec026b',
      "Ciana thought the other girls at the dorm were bullies.\nThe other girls don't bully her anymore.\nSo she kissed one of them on the lips.\nso she bullied them.", 'b',
      'Neither intervening action reliably explains why the bullying stopped; the preference needs source adjudication.'],
  ].map(([id, text, annotatedLabel, reason]) => ({ dataset: 'anli', id: id!, aliases: [], text: text!,
    annotatedLabel: annotatedLabel!, reason: reason!, status: 'pending' as const })),
  {
    dataset: 'folio', id: 'story:337', aliases: ['folio:story:337', 'story337'],
    text: 'Jim is either not a professional basketball player or not a slow runner.',
    annotatedLabel: 'Knicks=False; not Knicks=True; athlete=Unknown', status: 'pending',
    reason: 'The English disjunction permits Jim to be a fast professional/Knicks player. Source formalization instead uses negated XOR of positive predicates, which forces different conclusions.',
  },
  {
    dataset: 'folio', id: 'story:162', aliases: ['folio:story:162', 'story162'],
    text: 'Peter was invited to play piano at the concert hall.',
    annotatedLabel: 'Oliver piano=False; Oliver violin=Unknown; Peter good at piano=True', status: 'pending',
    reason: 'Source formalization turns an invitation into actual concert performance; the model-visible English does not establish that antecedent.',
  },
  {
    dataset: 'folio', id: 'story:56', aliases: ['folio:story:56', 'story56'],
    text: 'If a person is the leader of a country for life, that person is in a monarchy.',
    annotatedLabel: 'Elizabeth king=False; monarchy=True; female in monarchy=True', status: 'pending',
    reason: 'Source formalization drops the country and for-life conditions, making any leader a monarchy member; the English does not establish those conditions for Elizabeth.',
  },
  {
    dataset: 'folio', id: 'story:8', aliases: ['folio:story:8', 'story8'],
    text: 'Miroslav published a book in 1946.',
    annotatedLabel: 'Czech person wrote a book in 1946=True', status: 'pending',
    reason: 'Publication in 1946 does not establish that the person wrote the book in 1946; the conclusion requires source adjudication.',
  },
  {
    dataset: 'folio', id: 'story:348', aliases: ['folio:story:348', 'story348'],
    text: 'Someone is either a Yale student or a Harvard student.',
    annotatedLabel: 'Susan college=Unknown; diet and diligent=True; no diet and diligent=False', status: 'pending',
    reason: 'The source formalization universally quantifies an English existential. Someone being a student does not establish that Susan is a student.',
  },
  {
    dataset: 'folio', id: 'story:377', aliases: ['folio:story:377', 'story377'],
    text: 'People eat meat regularly or are vegetation.',
    annotatedLabel: 'Jeremy busy=Unknown; busy or enjoys meat=True; conditional=False', status: 'pending',
    reason: 'The source formalization interprets vegetation as vegetarian, joining it to another premise; the visible literal term does not establish that relation.',
  },
  {
    dataset: 'folio', id: 'story:416', aliases: ['folio:story:416', 'story416'],
    text: 'James is either good at planning or good at math.',
    annotatedLabel: 'planning=Unknown; planning or math=False; chemistry or math=False; conditional=True', status: 'pending',
    reason: 'Math(james) is unconstrained: with chemistry, award, experiment and planning false, either value of Math satisfies the premises (a different student witnesses the existential). Both disputed conclusions therefore admit true and false models rather than being contradicted.',
  },
  {
    dataset: 'kqapro', id: 'train:33143', aliases: ['train_33143'],
    text: 'What was the work that Robert Wise was nominated for the Academy Award for Best Film Editing about?',
    annotatedLabel: '14th Academy Awards', status: 'pending',
    reason: 'The English asks about the nominated work, but the original program reads the nomination statement-is-subject-of qualifier (the award ceremony). The KB separately gives Citizen Kane as for-work. The wording does not unambiguously request the ceremony.',
  },
  {
    dataset: 'folio', id: 'story:477', aliases: ['folio:story:477', 'story477'],
    text: 'All software is programmed. An APP is either related to YouTube or Instagram.',
    annotatedLabel: 'program=True; good or program=True; not related and program=False; related or program=False', status: 'pending',
    reason: 'The English does not state that TikTok is an APP, or equate programmed with a program. Its generic APP premise also leaves quantification ambiguous. Those missing connections and the exclusive-or reading cannot be silently supplied to force the source labels.',
  },
  {"dataset": "banking77", "id": "2731b69194df28fe70ad364791dae88fb7b65a636497ec95184c88edecf1c371", "aliases": ["c1651b366ec04d2d063681139d367182eb8285ca65b5851f9838ff474e54bcfe", "e12b67c6290668b4f27f032c6f3fe12475a73ae73ef14095912fdcf3adbaeff5"], "text": "How long will the $1 charge show as pending?", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "A pending one-dollar charge does not establish an extra charge rather than an ordinary pending payment or authorization hold."},
  {"dataset": "banking77", "id": "ff318b6e0c43f335f5304e69cb9b634b72601d10efeffe481c26b6864813f58e", "aliases": ["ddb2bfee38596136545d3d1055957372047ddac3c60d91eb7398f2a4ba5521fb", "ee834eb9a3074e25cf06db1bca927058239df2ad48cac43362294b36e559cf83"], "text": "I wanted to know about a pending dollar that is on my statement.", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "The amount and pending status alone do not establish an extra charge."},
  {"dataset": "banking77", "id": "af4146bc4ea82ac4f160e12c0f389e515017ff70b0b1b930edea33ddbc4f2e28", "aliases": ["2dafc63b62286f329ca9eb49cafbd6cc608ac38922253a4e2536e74ce8ea676f", "62d1e4c5b4559538fe864cb457c6ca1a0988fed0e94e11b56bbf3375c371997f"], "text": "I want a refund on a direct debit.", "annotatedLabel": "direct_debit_payment_not_recognised", "status": "pending", "reason": "Requesting a refund does not establish that a direct debit was unrecognized, duplicated or an extra charge."},
  {"dataset": "folio", "id": "story:454", "aliases": ["folio:story:454", "story454"], "text": "Either an animal can swim or it can walk. Liam is either an animal that can walk and enjoys water, or is neither an animal that can walk nor enjoys water.", "annotatedLabel": "Liam enjoys sun and splashing=True; neither=False", "status": "pending", "reason": "Source FOL drops animal restrictions, quantifying swim-or-walk and the animal rules over every individual. English permits the negative Liam branch without establishing he is an animal; a non-animal Liam with no walking, water, sun or splashing is a countermodel to the required positive conclusion."},
  {"dataset": "banking77", "id": "1d278f3fadf8451f95722831557bdd02cf17eaf9f1ebecd72624fd3b8e401757", "aliases": ["e8a776f03ae287cc587f56138444a2107d927ea93a4cff8d1980eeac0e975db8", "e0a860955d9ac37728a6cf4343616d91199bb6926fb7ef284c68bb2048578f0e"], "text": "I have been charged a pound for something that appears on my statement", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "Being charged a pound for something on a statement does not establish that the charge is additional, duplicated or unrecognized; the extra-charge label needs review."},
  {"dataset": "folio", "id": "story:423", "aliases": ["folio:story:423", "story423"], "text": "Students either go to the park or go to the movies. James does not have class during the weekend.", "annotatedLabel": "James summer camp=Unknown; park or summer camp=True", "status": "pending", "reason": "The English rules apply to students, but nothing establishes that James is a student. The source FOL removes the student guards and applies those rules to everyone. A non-student James with no class, no park and no camp satisfies the English premises but falsifies the disputed conclusion."},
  {"dataset": "banking77", "id": "e24a8c9b2816e9d5f78e1fc1f91478a3c818fb35ffc7289f744612dd3438fea3", "aliases": ["20b9c148712b7077905a8c43f53462db3e5f511cebee81bf01f43b9983ad30b0", "7bdc372ff6191fca4fbba8e42affdfe74589846c69819375568ae4462e0653cd"], "text": "What is the dollar that I have pending on my statement there?", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "Pending status and a dollar amount do not establish an additional or unrecognized charge."},
  {"dataset": "banking77", "id": "73a31e629a8e9b26628b1ceba0bbd7098dcf1b4bc97ede7894020a7545ef0432", "aliases": ["2adfb75cf940d7bdbf374ea4049c77badf63227a821b5b7dfacbd7d28da3e206", "2d3886d492d6e86573c2db0c370f6d84d65da188536467de3735ec67ecb3203d"], "text": "Why is it showing that my account has been charged a dollar that is showing as pending?", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "Pending status and a dollar amount do not establish an additional or unrecognized charge."},
  {"dataset": "folio", "id": "story:409", "aliases": ["folio:story:409", "story409"], "text": "No fruits that are beneficial to people are on a warning list. All fruits with the color red contain a large amount of vitamin C.", "annotatedLabel": "K vitamin C or warning=True; warning or red=True", "status": "pending", "reason": "English fruit guards cannot be dropped: non-fruit K that is beneficial but not apple, red, vitamin-rich or warned satisfies every premise and falsifies both asserted disjunctions."},
  {"dataset": "entailmentbank", "id": "LEAP__7_10338", "aliases": ["entailmentbank:LEAP__7_10338", "LEAP_7_10338"], "text": "an example of a behavioral adaptation is a bird building a nest in the warm ash from a volcano", "annotatedLabel": "supported", "status": "pending", "reason": "The stored proof reverses subtype implications: inherited behavior does not establish instinctive behavior, and positive survival impact does not establish behavioral adaptation. The English source facts do not supply either missing direction."},
  {"dataset": "banking77", "id": "020e5ee369757c51f6efd631c3aa5ca788bdb8bd6eec9e187d2b2afc8daa2c13", "aliases": ["1cf4e3a79906be39f357ff026eea582cf3cc7f11c60e5d16b4fbfc54f1a3ea68", "9fa1fbf1610ca6d354544ad573b6eafba2ee03a9afe75f9f34daf63fcee48f84"], "text": "Why was my account assessed a fee?", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "The wording does not establish that a fee is extra or that an amount/reversal question concerns an unrecognized transaction. Preserve the source annotation pending independent review."},
  {"dataset": "banking77", "id": "3189b7e044ef369b31538dbedd8ba6181e4d117cc222dcdc7043b72f0877341d", "aliases": ["3d8d27477b32dfc6f0423d394f304995fef2b50765eaec747901384c6b7729f6", "cf0c53b2453a143f8d5bd9c7cd6f28c1d7240cf196a1cb2186c6a14fcdcdcad5"], "text": "Please help me figure out the reason for the odd withdrawal amount from my account?", "annotatedLabel": "cash_withdrawal_not_recognised", "status": "pending", "reason": "The wording does not establish that a fee is extra or that an amount/reversal question concerns an unrecognized transaction. Preserve the source annotation pending independent review."},
  {"dataset": "banking77", "id": "0df99142c8ffb7e4d1175312711bf6f98f27ec5cad089e47e282fed99488301d", "aliases": ["1613ab3719afdb323664de7aa4fc34aac1c6c9ab7a48d9b9beec730afc3cb2bd", "f1bdfbfa8688e1607f166463a756113aefaa2fe59f0d53e65902ba2f89e16a60"], "text": "What was the $1 charge for on my statement?", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "The wording does not establish that a fee is extra or that an amount/reversal question concerns an unrecognized transaction. Preserve the source annotation pending independent review."},
  {"dataset": "banking77", "id": "53d18a426a85d64ac1f7dd1ea464b8b2189a6d1f7242b8708f620b61d68c6c7a", "aliases": ["73c96e914b511700d4011426efbab7ad20ccb5b1193a22e1bbc4166fcb1b2e3d", "e965dee223bca285a51e784fe1c9603ee7e1f8ae012ae23882fcd8dd8c679a0a"], "text": "The $1.00 has still not been reverted as indicated in the previous email.", "annotatedLabel": "extra_charge_on_statement", "status": "pending", "reason": "The wording does not establish that a fee is extra or that an amount/reversal question concerns an unrecognized transaction. Preserve the source annotation pending independent review."},
  {"dataset": "folio", "id": "story:417", "aliases": ["folio:story:417", "story417"], "text": "L-2021 is either in the library or produced by LG.", "annotatedLabel": "False", "status": "pending", "reason": "The source treats either-or as exclusive. A library monitor produced by both AOC and LG without type-c satisfies all premises and makes the inclusive English disjunction true. English also omits a monitor premise for L-2021."},
  {"dataset": "folio", "id": "story:378", "aliases": ["folio:story:378", "story378"], "text": "Carol is not getting married or has friends who are getting married.", "annotatedLabel": "Carol outgoing=False; neither celebrating nor outgoing=True", "status": "pending", "reason": "The source negates the whole merged getting-married-or-friends predicate, whereas English can negate getting married alone. Carol with married friends, outgoing, enjoying celebrations and attending weddings satisfies the literal English but falsifies the source conclusions. The negation scope needs adjudication."},
  {"dataset": "folio", "id": "story:422", "aliases": ["folio:story:422", "story422"], "text": "Customers either subscribe to AMC service or HBO service. James watches TV series in cinemas.", "annotatedLabel": "weekly cinema or three movies=True; availability or TV=True", "status": "pending", "reason": "No English premise establishes James as a customer; the source formalization removes customer guards. A non-customer James watching TV in cinemas but without subscriptions or three-movie access satisfies the premises and falsifies the asserted entailments. Availability also does not establish actual weekly watching."},
];

/** Stable visible identity also catches legacy batches that omitted dataset_records. */
export function anliReviewText(story: Record<string, unknown>): string | undefined {
  const fields = ['beginning', 'ending', 'a', 'b'].map(key => story[key]);
  return fields.every(value => typeof value === 'string') ? fields.join('\n') : undefined;
}

export function pendingSourceReview(dataset: string, id: string): SourceReview | undefined {
  return SOURCE_REVIEWS.find(review => review.status === 'pending' && review.dataset === dataset &&
    (review.id === id || review.aliases.includes(id)));
}

/** A whole task is held when any of its original source records needs review. */
export function sourceReviewReason(record: Record<string, unknown>):
    string | undefined {
  const curriculum = record.curriculum as { family?: string; shape?: string } | undefined;
  const inferred = curriculum?.family === 'entailment_premises' ? 'entailmentbank' : curriculum?.family === 'kqapro_question' ? 'kqapro' : curriculum?.family === 'anli_batch' ? 'anli' :
    ['folio_batch', 'folio_entailment'].includes(curriculum?.family ?? '') ? 'folio' : undefined;
  const dataset = record.dataset ?? inferred ?? record.source;
  if (typeof dataset !== 'string') return undefined;
  if (dataset === 'anli') {
    const semantics = record.semantics as { inputs?: { stories?: unknown[] } } | undefined;
    const texts = semantics?.inputs?.stories?.flatMap(story => story && typeof story === 'object' ?
      [anliReviewText(story as Record<string, unknown>)] : []) ?? [];
    if (SOURCE_REVIEWS.some(review => review.dataset === dataset && review.status === 'pending' && texts.includes(review.text)))
      return 'source_review_pending';
  }
  const ids = [record.dataset_records, record.source_ids, ['folio', 'entailmentbank'].includes(dataset) ? record.source_groups : []]
    .flatMap(value => Array.isArray(value) ? value : []);
  if (['folio', 'kqapro', 'entailmentbank'].includes(dataset) && curriculum?.shape) ids.push(curriculum.shape);
  return ids.some(id => typeof id === 'string' && pendingSourceReview(dataset, id)) ?
    'source_review_pending' : undefined;
}
