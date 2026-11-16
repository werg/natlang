var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "nsta_req": {
    "name": "nsta_req",
    "structName": "glue_msg_native_state_req",
    "className": "GlueMsgNativeStateReq",
    "fields": [
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "nsta_res": {
    "name": "nsta_res",
    "structName": "glue_msg_native_state_res",
    "className": "GlueMsgNativeStateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "meta_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12080-401f8ad";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`, `nativeStateStore` means `putNativeState` / `getNativeState` are available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true,
      nativeStateStore: true
    });
    __publicField(this, "slotFileId", 0);
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // note: we overlay instead of using llama-server default_template_kwargs, because we cannot transfer complex data structure via GLUE
    // overlay allow mixed data type or nested structure for kwargs
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
    });
  }
  /**
   * Store a natlang native state (natlang's store-file format, natlang.native-state/1) after the model is loaded, as the
   * server's PUT /v1/neuralese/native-states/:id does: the bytes are verified (layout, finite values, content ID), kept as
   * received and, for a state of this model's identity, registered so prompts consume it exactly.
   * @param state The state's bytes
   * @param id The ID the state must have (its content ID); omit to accept any
   * @returns The consumption mode and the state's metadata ({id, ...metadata})
   */
  putNativeState(state, id = "") {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        const result = yield this.proxy.wllamaAction(
          "native_state",
          { _name: "nsta_req", action: "put", id, filename }
        );
        if (!result.success) {
          throw new WllamaError("Failed to put native state", "inference_error");
        }
        return yield this.getSingleResult(result.req_id, "native state put");
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Read a stored native state (one this engine produced, or one put or loaded), as the server's
   * GET /v1/neuralese/native-states/:id does: its bytes exactly as stored, natlang's store-file format.
   * @param id The state's ID
   * @returns The state's bytes and its metadata ({id, ...metadata})
   */
  getNativeState(id) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const result = yield this.proxy.wllamaAction(
        "native_state",
        { _name: "nsta_req", action: "get", id, filename }
      );
      if (!result.success) {
        throw new WllamaError("Failed to get native state", "inference_error");
      }
      const state = yield this.proxy.slotFileRead(filename);
      return { state, meta: JSON.parse(result.meta_json) };
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
